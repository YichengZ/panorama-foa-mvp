from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from panorama_foa.ambisonics.exporter import export_foa_wav, write_metadata
from panorama_foa.ambisonics.foa import encode_global_ambience, encode_regional_foa
from panorama_foa.ambisonics.mixer import mix_foa_layers
from panorama_foa.audio.decode import decode_to_mono_wav
from panorama_foa.audio.mock import MockTextToAudioProvider
from panorama_foa.audio.processing import TARGET_SAMPLE_RATE, process_mono_audio
from panorama_foa.audio.provider_base import TextToAudioProvider
from panorama_foa.coordinates import normalized_panorama_to_angles
from panorama_foa.planner.base import ScenePlanner
from panorama_foa.planner.manual import ManualPlanPlanner
from panorama_foa.schemas import ScenePlan


@dataclass
class PipelineResult:
    output_dir: Path
    plan: ScenePlan
    wav_path: Path
    metadata_path: Path


def validate_and_filter_plan(plan: ScenePlan, *, duration_seconds: float, max_sources: int) -> ScenePlan:
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
        if len(kept) >= max_sources:
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
        )
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
            )
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
        export_foa_wav(wav_path, mix, sample_rate=TARGET_SAMPLE_RATE)
        write_metadata(metadata_path, duration_seconds=plan.duration_seconds, yaw_offset_deg=self.yaw_offset_deg)
        self._verify_wav(wav_path, plan.duration_seconds)
        return PipelineResult(output_dir=output_dir, plan=plan, wav_path=wav_path, metadata_path=metadata_path)

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
    ):
        if self.force or not raw_path.exists():
            self.audio_provider.generate(
                prompt=prompt,
                duration_seconds=duration_seconds,
                loop=loop,
                output_path=raw_path,
            )
        readable_path = raw_path
        if raw_path.suffix.lower() not in {".wav", ".flac", ".aiff", ".aif"}:
            decoded_path = stem_path.with_name(f"{stem_path.stem}.decoded.wav")
            decode_to_mono_wav(raw_path, decoded_path)
            readable_path = decoded_path
        data, sample_rate = sf.read(readable_path, dtype="float32", always_2d=False)
        stem = process_mono_audio(
            data,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            gain_db=gain_db,
            source_id=source_id,
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
