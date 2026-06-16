from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

from sonoworld.core.artifacts import ref, save_understanding
from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.schemas.common import FileRef, write_json
from sonoworld.schemas.understanding import GlobalSound, SceneUnderstanding, SoundObject
from sonoworld.utils.audio_source_utils import normalize_audio_source_type, normalize_peak_db
from sonoworld.utils.text_utils import clean_text


DEFAULT_PROMPT_PATH = "configs/prompts/default_uncond.txt"
DEFAULT_MODEL = "gpt-4.1"


class GPTUnderstandingStage(Stage):
    """OpenAI GPT-backed panorama understanding stage.

    Third-party imports are kept inside the API call so importing this module
    stays cheap in pipelines that do not use GPT.
    """

    name = "understanding"
    backend = "gpt"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_path: str = DEFAULT_PROMPT_PATH,
        max_retries: int = 3,
        grounding_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.prompt_path = prompt_path
        self.max_retries = max(1, int(max_retries))
        self.grounding_threshold = float(grounding_threshold)

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        stage_cfg = ctx.stage_config(self.name)

        model = str(stage_cfg.get("model", self.model))
        prompt_path = self._resolve_prompt_path(
            stage_cfg.get("prompt_path", self.prompt_path),
            scene_root=scene_root,
        )
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {prompt_path}")

        panorama = paths.find_panorama()
        response_text = self._request_json(
            model=model,
            prompt=prompt,
            image_path=panorama,
            max_retries=int(stage_cfg.get("max_retries", self.max_retries)),
        )
        raw_items = self._parse_response_items(response_text)
        understanding = self._to_understanding(
            raw_items,
            model=model,
            prompt_path=prompt_path,
            panorama=panorama,
            response_text=response_text,
        )

        summary_path = save_understanding(paths, understanding)

        grounding_threshold = float(
            stage_cfg.get("grounding_threshold", self.grounding_threshold)
        )
        if grounding_threshold != 0.7:
            write_json(paths.grounding_input, understanding.to_grounding_input(grounding_threshold))

        legacy_path = paths.understanding / "understanding_legacy.json"

        return StageResult(
            status="done",
            inputs={
                "panorama": ref(panorama, scene_root, role="panorama", media_type="image"),
                "prompt": FileRef.from_path(
                    prompt_path,
                    scene_root=scene_root,
                    role="understanding_prompt",
                    media_type="text/plain",
                ),
            },
            outputs={
                "summary": ref(
                    summary_path,
                    scene_root,
                    role="understanding_metadata",
                    media_type="application/json",
                ),
                "legacy": ref(
                    legacy_path,
                    scene_root,
                    role="understanding_legacy",
                    media_type="application/json",
                ),
                "grounding_input": ref(
                    paths.grounding_input,
                    scene_root,
                    role="grounding_input",
                    media_type="application/json",
                ),
            },
            message="GPT understanding complete.",
            metadata={
                "backend": self.backend,
                "model": model,
                "num_items": len(raw_items),
                "prompt_path": str(prompt_path),
            },
        )

    def _request_json(
        self,
        model: str,
        prompt: str,
        image_path: Path,
        max_retries: int,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "GPTUnderstandingStage requires the OpenAI Python package. "
                "Install `openai` and set OPENAI_API_KEY before running this stage."
            ) from exc

        client = OpenAI()
        image_url = self._image_data_url(image_path)
        last_response = ""
        max_attempts = max(1, int(max_retries))

        for attempt in range(1, max_attempts + 1):
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
            )
            last_response = (result.choices[0].message.content or "").strip()
            try:
                self._parse_response_items(last_response)
                return last_response
            except (TypeError, ValueError, json.JSONDecodeError):
                if attempt == max_attempts:
                    break

        raise ValueError(
            "GPT did not return valid understanding JSON after "
            f"{max_attempts} attempt(s). Last response: {last_response[:500]}"
        )

    def _to_understanding(
        self,
        raw_items: List[Dict[str, Any]],
        model: str,
        prompt_path: Path,
        panorama: Path,
        response_text: str,
    ) -> SceneUnderstanding:
        objects: List[SoundObject] = []
        global_sounds: List[GlobalSound] = []

        for index, item in enumerate(raw_items):
            grounding_label = (
                clean_text(item.get("grounding_label"))
                or clean_text(item.get("label"))
                or "global"
            )
            source_type = normalize_audio_source_type(item.get("source_type"), grounding_label)
            diffusion_prompt = clean_text(item.get("diffusion_prompt"))
            peak_db = normalize_peak_db(item.get("peak_db"), source_type)

            if grounding_label.lower() == "global" or source_type == "background":
                global_sounds.append(
                    GlobalSound(
                        label="global",
                        diffusion_prompt=diffusion_prompt or "quiet scene ambience",
                        source_type="background",
                        peak_db=peak_db,
                        metadata={"raw_index": index},
                    )
                )
                continue

            label = clean_text(item.get("label")) or grounding_label
            objects.append(
                SoundObject(
                    label=label,
                    grounding_label=grounding_label,
                    diffusion_prompt=diffusion_prompt or f"{grounding_label} sound",
                    source_type=source_type,
                    peak_db=peak_db,
                    metadata={"raw_index": index},
                )
            )

        if not global_sounds:
            global_sounds.append(
                GlobalSound(
                    diffusion_prompt="quiet scene ambience",
                    peak_db=-30.0,
                    metadata={"created_by": self.backend},
                )
            )

        return SceneUnderstanding(
            scene_description="Sound sources inferred from the panorama.",
            objects=objects,
            global_sounds=global_sounds,
            backend=self.backend,
            model=model,
            metadata={
                "prompt_path": str(prompt_path),
                "panorama": str(panorama),
                "raw_response": response_text,
            },
        )

    def _parse_response_items(self, response_text: str) -> List[Dict[str, Any]]:
        text = self._strip_code_fence(response_text)
        data = json.loads(text)

        if isinstance(data, dict):
            if isinstance(data.get("items"), list):
                data = data["items"]
            elif isinstance(data.get("sounds"), list):
                data = data["sounds"]
            else:
                data = [data]

        if not isinstance(data, list):
            raise TypeError("GPT response must be a JSON array or an object containing one.")

        items: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Every GPT understanding item must be a JSON object.")
            items.append(item)

        if not items:
            raise ValueError("GPT response contained no understanding items.")

        return items

    def _resolve_prompt_path(self, prompt_path: str | Path, scene_root: Path) -> Path:
        path = Path(prompt_path)
        if path.is_absolute():
            return path

        candidates = [
            scene_root / path,
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[-1]

    def _image_data_url(self, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _strip_code_fence(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
