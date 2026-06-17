from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from panorama_foa.ambisonics.exporter import export_foa_wav, write_metadata
from panorama_foa.ambisonics.foa import encode_global_ambience, encode_regional_foa
from panorama_foa.ambisonics.mixer import mix_foa_layers
from panorama_foa.audio.decode import decode_to_mono_wav
from panorama_foa.audio.mock import MockTextToAudioProvider
from panorama_foa.audio.processing import TARGET_SAMPLE_RATE, process_mono_audio, requested_generation_duration
from panorama_foa.audio.provider_base import TextToAudioProvider
from panorama_foa.coordinates import normalized_panorama_to_angles
from panorama_foa.planner.base import ScenePlanner
from panorama_foa.planner.manual import ManualPlanPlanner
from panorama_foa.schemas import MAX_REGIONAL_SOURCES, ScenePlan


@dataclass
class PipelineResult:
    output_dir: Path
    plan: ScenePlan
    wav_path: Path
    metadata_path: Path
    loopcheck_path: Path
    metrics_path: Path


def validate_and_filter_plan(plan: ScenePlan, *, duration_seconds: float, max_sources: int) -> ScenePlan:
    source_limit = max(0, min(int(max_sources), MAX_REGIONAL_SOURCES))
    kept = []
    seen_labels = set()
    for source in sorted(plan.regional_sources, key=lambda item: item.confidence, reverse=True):
        if source.confidence < 0.60:
            continue
        label_key = source.label.strip().lower()
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        kept.append(source)
        if len(kept) >= source_limit:
            break
    data = plan.model_dump()
    data["duration_seconds"] = duration_seconds
    data["regional_sources"] = kept
    return ScenePlan.model_validate(data)


class PanoramaToFOAPipeline:
    def __init__(
        self,
        *,
        planner: ScenePlanner,
        audio_provider: TextToAudioProvider,
        yaw_offset_deg: float = 0.0,
        force: bool = False,
    ) -> None:
        self.planner = planner
        self.audio_provider = audio_provider
        self.yaw_offset_deg = yaw_offset_deg
        self.force = force

    def run(
        self,
        *,
        panorama_path: Path,
        output_dir: Path,
        duration_seconds: float,
        scene_description: str | None = None,
        max_sources: int = 5,
        allow_speech: bool = False,
        allow_music: bool = False,
    ) -> PipelineResult:
        _validate_duration(duration_seconds)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = output_dir / "panorama_analysis.jpg"
        self._copy_analysis_image(panorama_path, analysis_path)

        plan = self.planner.plan(
            panorama_path=panorama_path,
            duration_seconds=duration_seconds,
            scene_description=scene_description,
            max_sources=max_sources,
            allow_speech=allow_speech,
            allow_music=allow_music,
        )
        plan = validate_and_filter_plan(plan, duration_seconds=duration_seconds, max_sources=max_sources)
        (output_dir / "scene_plan.json").write_text(plan.model_dump_json(indent=2) + "\n")

        layers = []
        metrics: dict[str, object] = {
            "schema_version": "1.0",
            "base_loop_duration_seconds": float(plan.duration_seconds),
            "loopcheck_repeat_count": 2,
            "loopcheck_duration_seconds": float(plan.duration_seconds * 2),
            "target_rms_dbfs": -30.0,
            "peak_ceiling_dbfs": -3.0,
            "raw_audio": {},
            "stems": {},
        }
        raw_dir = output_dir / "raw_audio"
        stems_dir = output_dir / "stems"
        raw_dir.mkdir(exist_ok=True)
        stems_dir.mkdir(exist_ok=True)

        global_stem = self._generate_stem(
            source_id=plan.global_ambience.id,
            prompt=plan.global_ambience.prompt,
            loop=plan.global_ambience.loop,
            gain_db=plan.global_ambience.gain_db,
            duration_seconds=plan.duration_seconds,
            raw_path=raw_dir / f"global_ambience{self._raw_extension()}",
            stem_path=stems_dir / "global_ambience.wav",
            metrics=metrics,
        )
        metrics["stems"][plan.global_ambience.id] = _audio_metrics(global_stem, TARGET_SAMPLE_RATE)
        layers.append(encode_global_ambience(global_stem))

        for index, source in enumerate(plan.regional_sources):
            raw_name = f"source_{index:02d}{self._raw_extension()}"
            stem_name = f"source_{index:02d}.wav"
            stem = self._generate_stem(
                source_id=source.id,
                prompt=source.prompt,
                loop=source.loop,
                gain_db=source.gain_db,
                duration_seconds=plan.duration_seconds,
                raw_path=raw_dir / raw_name,
                stem_path=stems_dir / stem_name,
                metrics=metrics,
            )
            metrics["stems"][source.id] = _audio_metrics(stem, TARGET_SAMPLE_RATE)
            azimuth, elevation = normalized_panorama_to_angles(
                source.center_u,
                source.center_v,
                self.yaw_offset_deg,
            )
            layers.append(
                encode_regional_foa(
                    stem,
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    spread=source.spread,
                )
            )

        mix = mix_foa_layers(layers)
        wav_path = output_dir / "scene_foa.wav"
        metadata_path = output_dir / "scene_foa.metadata.json"
        loopcheck_path = output_dir / "scene_foa_loopcheck.wav"
        metrics_path = output_dir / "scene_foa.metrics.json"
        export_foa_wav(wav_path, mix, sample_rate=TARGET_SAMPLE_RATE)
        export_foa_wav(loopcheck_path, np.tile(mix, (2, 1)), sample_rate=TARGET_SAMPLE_RATE)
        write_metadata(metadata_path, duration_seconds=plan.duration_seconds, yaw_offset_deg=self.yaw_offset_deg)
        metrics["final"] = _audio_metrics(mix, TARGET_SAMPLE_RATE)
        metrics["loopcheck"] = _audio_metrics(np.tile(mix, (2, 1)), TARGET_SAMPLE_RATE)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        self._verify_wav(wav_path, plan.duration_seconds)
        self._verify_wav(loopcheck_path, plan.duration_seconds * 2)
        return PipelineResult(
            output_dir=output_dir,
            plan=plan,
            wav_path=wav_path,
            metadata_path=metadata_path,
            loopcheck_path=loopcheck_path,
            metrics_path=metrics_path,
        )

    def _generate_stem(
        self,
        *,
        source_id: str,
        prompt: str,
        loop: bool,
        gain_db: float,
        duration_seconds: float,
        raw_path: Path,
        stem_path: Path,
        metrics: dict[str, object],
    ) -> np.ndarray:
        generation_duration = requested_generation_duration(duration_seconds, loop=loop)
        if self.force or not raw_path.exists() or not _raw_has_required_duration(raw_path, generation_duration):
            self.audio_provider.generate(
                prompt=prompt,
                duration_seconds=generation_duration,
                loop=loop,
                output_path=raw_path,
            )
        readable_path = raw_path
        if raw_path.suffix.lower() not in {".wav", ".flac", ".aiff", ".aif"}:
            decoded_path = stem_path.with_name(f"{stem_path.stem}.decoded.wav")
            decode_to_mono_wav(raw_path, decoded_path)
            readable_path = decoded_path
        data, sample_rate = sf.read(readable_path, dtype="float32", always_2d=False)
        metrics["raw_audio"][source_id] = _audio_metrics(data, sample_rate)
        stem = process_mono_audio(
            data,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            gain_db=gain_db,
            source_id=source_id,
            loop=loop,
        )
        sf.write(stem_path, stem, TARGET_SAMPLE_RATE, subtype="PCM_16")
        if readable_path != raw_path:
            readable_path.unlink(missing_ok=True)
        return stem

    def _raw_extension(self) -> str:
        extension = getattr(self.audio_provider, "raw_extension", ".wav")
        if not str(extension).startswith("."):
            return f".{extension}"
        return str(extension)

    def _copy_analysis_image(self, panorama_path: Path, analysis_path: Path) -> None:
        try:
            from panorama_foa.image_utils import create_analysis_image, validate_panorama_image

            validate_panorama_image(panorama_path)
            create_analysis_image(panorama_path, analysis_path)
        except ModuleNotFoundError:
            shutil.copyfile(panorama_path, analysis_path)

    def _verify_wav(self, wav_path: Path, duration_seconds: float) -> None:
        info = sf.info(wav_path)
        if info.channels != 4 or info.samplerate != TARGET_SAMPLE_RATE:
            raise RuntimeError(f"invalid FOA output: {info}")
        if info.frames != round(duration_seconds * TARGET_SAMPLE_RATE):
            raise RuntimeError(f"unexpected FOA frame count: {info.frames}")


def build_mock_manual_pipeline(plan_path: Path, *, yaw_offset_deg: float = 0.0, force: bool = False) -> PanoramaToFOAPipeline:
    return PanoramaToFOAPipeline(
        planner=ManualPlanPlanner(plan_path),
        audio_provider=MockTextToAudioProvider(),
        yaw_offset_deg=yaw_offset_deg,
        force=force,
    )


def _validate_duration(duration_seconds: float) -> None:
    if not 0.5 <= float(duration_seconds) <= 30.0:
        raise ValueError("duration_seconds must be between 0.5 and 30.0 seconds")


def _audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        mono = array
        channels = 1
    else:
        channels = int(array.shape[1])
        mono = np.mean(array, axis=1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    rms = float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2))) if array.size else 0.0
    frames = int(array.shape[0]) if array.ndim else 0
    edge_samples = min(round(0.1 * sample_rate), frames // 2)
    endpoint_jump = float(abs(mono[0] - mono[-1])) if frames else 0.0
    if edge_samples:
        edge_diff = mono[:edge_samples] - mono[-edge_samples:]
        edge_rms = float(np.sqrt(np.mean(np.asarray(edge_diff, dtype=np.float64) ** 2)))
    else:
        edge_rms = 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "frames": frames,
        "peak": peak,
        "peak_dbfs": _dbfs(peak),
        "rms": rms,
        "rms_dbfs": _dbfs(rms),
        "endpoint_jump": endpoint_jump,
        "edge_100ms_diff_dbfs": _dbfs(edge_rms),
    }


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _raw_has_required_duration(path: Path, duration_seconds: float) -> bool:
    if path.suffix.lower() not in {".wav", ".flac", ".aiff", ".aif"}:
        return True
    try:
        info = sf.info(path)
    except sf.LibsndfileError:
        return False
    return info.frames >= round(duration_seconds * info.samplerate)
