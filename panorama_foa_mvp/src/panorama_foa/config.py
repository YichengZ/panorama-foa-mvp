from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_VISION_MODEL_ENV = "OPENAI_VISION_MODEL"
ELEVENLABS_API_KEY_ENV = "ELEVENLABS_API_KEY"
ELEVENLABS_SOUND_MODEL_ENV = "ELEVENLABS_SOUND_MODEL"
ELEVENLABS_PROMPT_INFLUENCE_ENV = "ELEVENLABS_PROMPT_INFLUENCE"
MMAUDIO_MODEL_VARIANT_ENV = "MMAUDIO_MODEL_VARIANT"
MMAUDIO_MODEL_PATH_ENV = "MMAUDIO_MODEL_PATH"
MMAUDIO_DEVICE_ENV = "MMAUDIO_DEVICE"
MMAUDIO_STEPS_ENV = "MMAUDIO_STEPS"
MMAUDIO_GUIDANCE_SCALE_ENV = "MMAUDIO_GUIDANCE_SCALE"
MMAUDIO_FULL_PRECISION_ENV = "MMAUDIO_FULL_PRECISION"
MMAUDIO_INFERENCE_MODE_ENV = "MMAUDIO_INFERENCE_MODE"
MMAUDIO_SONOWORLD_ROOT_ENV = "MMAUDIO_SONOWORLD_ROOT"

DEFAULT_OPENAI_VISION_MODEL = "gpt-4.1"
DEFAULT_ELEVENLABS_SOUND_MODEL = "eleven_text_to_sound_v2"
DEFAULT_ELEVENLABS_PROMPT_INFLUENCE = 0.4
DEFAULT_ELEVENLABS_TIMEOUT_SECONDS = 120.0
DEFAULT_ELEVENLABS_MAX_RETRIES = 3
DEFAULT_MMAUDIO_MODEL_VARIANT = "large_44k_v2"
DEFAULT_MMAUDIO_MODEL_PATH = ".cache/weights/mmaudio"
DEFAULT_MMAUDIO_STEPS = 25
DEFAULT_MMAUDIO_GUIDANCE_SCALE = 7.5
DEFAULT_MMAUDIO_FULL_PRECISION = False
DEFAULT_MMAUDIO_INFERENCE_MODE = "euler"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "panorama_scene_planner.txt"


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    api_key: str
    model: str = DEFAULT_OPENAI_VISION_MODEL
    prompt_path: Path = DEFAULT_PLANNER_PROMPT_PATH


@dataclass(frozen=True)
class ElevenLabsConfig:
    api_key: str
    model_id: str = DEFAULT_ELEVENLABS_SOUND_MODEL
    prompt_influence: float = DEFAULT_ELEVENLABS_PROMPT_INFLUENCE
    timeout_seconds: float = DEFAULT_ELEVENLABS_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_ELEVENLABS_MAX_RETRIES


@dataclass(frozen=True)
class MMAudioConfig:
    model_variant: str = DEFAULT_MMAUDIO_MODEL_VARIANT
    model_path: str = DEFAULT_MMAUDIO_MODEL_PATH
    device: str = ""
    steps: int = DEFAULT_MMAUDIO_STEPS
    guidance_scale: float = DEFAULT_MMAUDIO_GUIDANCE_SCALE
    full_precision: bool = DEFAULT_MMAUDIO_FULL_PRECISION
    inference_mode: str = DEFAULT_MMAUDIO_INFERENCE_MODE
    sonoworld_root: Path | None = None


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_vision_model: str
    elevenlabs_api_key: str
    elevenlabs_sound_model: str = DEFAULT_ELEVENLABS_SOUND_MODEL
    elevenlabs_prompt_influence: float = DEFAULT_ELEVENLABS_PROMPT_INFLUENCE
    elevenlabs_timeout_seconds: float = DEFAULT_ELEVENLABS_TIMEOUT_SECONDS
    elevenlabs_max_retries: int = DEFAULT_ELEVENLABS_MAX_RETRIES
    mmaudio_model_variant: str = DEFAULT_MMAUDIO_MODEL_VARIANT
    mmaudio_model_path: str = DEFAULT_MMAUDIO_MODEL_PATH
    mmaudio_device: str = ""
    mmaudio_steps: int = DEFAULT_MMAUDIO_STEPS
    mmaudio_guidance_scale: float = DEFAULT_MMAUDIO_GUIDANCE_SCALE
    mmaudio_full_precision: bool = DEFAULT_MMAUDIO_FULL_PRECISION
    mmaudio_inference_mode: str = DEFAULT_MMAUDIO_INFERENCE_MODE
    mmaudio_sonoworld_root: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        require_openai: bool = False,
        require_elevenlabs: bool = False,
    ) -> "Settings":
        load_dotenv_file()
        env = os.environ if environ is None else environ
        openai_api_key = _read_env(env, OPENAI_API_KEY_ENV)
        elevenlabs_api_key = _read_env(env, ELEVENLABS_API_KEY_ENV)
        if require_openai and not openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the OpenAI VLM planner.")
        if require_elevenlabs and not elevenlabs_api_key:
            raise ConfigurationError("ELEVENLABS_API_KEY is required for the ElevenLabs audio provider.")
        return cls(
            openai_api_key=openai_api_key,
            openai_vision_model=_read_env(env, OPENAI_VISION_MODEL_ENV)
            or DEFAULT_OPENAI_VISION_MODEL,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_sound_model=_read_env(env, ELEVENLABS_SOUND_MODEL_ENV)
            or DEFAULT_ELEVENLABS_SOUND_MODEL,
            elevenlabs_prompt_influence=_read_float_env(
                env,
                ELEVENLABS_PROMPT_INFLUENCE_ENV,
                DEFAULT_ELEVENLABS_PROMPT_INFLUENCE,
            ),
            mmaudio_model_variant=_read_env(env, MMAUDIO_MODEL_VARIANT_ENV)
            or DEFAULT_MMAUDIO_MODEL_VARIANT,
            mmaudio_model_path=_read_env(env, MMAUDIO_MODEL_PATH_ENV)
            or DEFAULT_MMAUDIO_MODEL_PATH,
            mmaudio_device=_read_env(env, MMAUDIO_DEVICE_ENV),
            mmaudio_steps=_read_int_env(env, MMAUDIO_STEPS_ENV, DEFAULT_MMAUDIO_STEPS),
            mmaudio_guidance_scale=_read_float_env(
                env,
                MMAUDIO_GUIDANCE_SCALE_ENV,
                DEFAULT_MMAUDIO_GUIDANCE_SCALE,
            ),
            mmaudio_full_precision=_read_bool_env(
                env,
                MMAUDIO_FULL_PRECISION_ENV,
                DEFAULT_MMAUDIO_FULL_PRECISION,
            ),
            mmaudio_inference_mode=_read_env(env, MMAUDIO_INFERENCE_MODE_ENV)
            or DEFAULT_MMAUDIO_INFERENCE_MODE,
            mmaudio_sonoworld_root=_read_optional_path(env, MMAUDIO_SONOWORLD_ROOT_ENV),
        )

    def openai_config(self) -> OpenAIPlannerConfig:
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the OpenAI VLM planner.")
        return OpenAIPlannerConfig(api_key=self.openai_api_key, model=self.openai_vision_model)

    def elevenlabs_config(self) -> ElevenLabsConfig:
        if not self.elevenlabs_api_key:
            raise ConfigurationError("ELEVENLABS_API_KEY is required for the ElevenLabs audio provider.")
        return ElevenLabsConfig(
            api_key=self.elevenlabs_api_key,
            model_id=self.elevenlabs_sound_model,
            prompt_influence=self.elevenlabs_prompt_influence,
            timeout_seconds=self.elevenlabs_timeout_seconds,
            max_retries=self.elevenlabs_max_retries,
        )

    def mmaudio_config(self) -> MMAudioConfig:
        return MMAudioConfig(
            model_variant=self.mmaudio_model_variant,
            model_path=self.mmaudio_model_path,
            device=self.mmaudio_device,
            steps=self.mmaudio_steps,
            guidance_scale=self.mmaudio_guidance_scale,
            full_precision=self.mmaudio_full_precision,
            inference_mode=self.mmaudio_inference_mode,
            sonoworld_root=self.mmaudio_sonoworld_root,
        )


class ConfigurationError(RuntimeError):
    """Raised when a paid API provider is selected without required config."""


def load_dotenv_file(dotenv_path: Path | None = None) -> None:
    """Load a .env file when python-dotenv is installed.

    Importing this module stays cheap for offline tests; missing python-dotenv
    is tolerated because real installs declare it as a project dependency.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    if dotenv_path is None:
        dotenv_path = PROJECT_ROOT / ".env"

    load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)


def openai_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
    prompt_path: Path | None = None,
    require_api_key: bool = True,
) -> OpenAIPlannerConfig:
    load_dotenv_file()
    env = os.environ if environ is None else environ
    api_key = _read_env(env, OPENAI_API_KEY_ENV)
    if require_api_key and not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is required for the OpenAI VLM planner. "
            "Set it in the environment or use the manual planner for offline runs."
        )

    return OpenAIPlannerConfig(
        api_key=api_key,
        model=_read_env(env, OPENAI_VISION_MODEL_ENV) or DEFAULT_OPENAI_VISION_MODEL,
        prompt_path=prompt_path or DEFAULT_PLANNER_PROMPT_PATH,
    )


def elevenlabs_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
    require_api_key: bool = True,
) -> ElevenLabsConfig:
    load_dotenv_file()
    env = os.environ if environ is None else environ
    api_key = _read_env(env, ELEVENLABS_API_KEY_ENV)
    if require_api_key and not api_key:
        raise ConfigurationError(
            "ELEVENLABS_API_KEY is required for the ElevenLabs audio provider. "
            "Set it in the environment or use the mock audio provider for offline runs."
        )

    return ElevenLabsConfig(
        api_key=api_key,
        model_id=_read_env(env, ELEVENLABS_SOUND_MODEL_ENV)
        or DEFAULT_ELEVENLABS_SOUND_MODEL,
        prompt_influence=_read_float_env(
            env,
            ELEVENLABS_PROMPT_INFLUENCE_ENV,
            DEFAULT_ELEVENLABS_PROMPT_INFLUENCE,
        ),
    )


def mmaudio_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
) -> MMAudioConfig:
    settings = Settings.from_env(environ=environ)
    return settings.mmaudio_config()


def redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[redacted]")


def _read_env(env: Mapping[str, str], name: str) -> str:
    return env.get(name, "").strip()


def _read_float_env(env: Mapping[str, str], name: str, default: float) -> float:
    value = _read_env(env, name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a floating-point number.") from exc


def _read_int_env(env: Mapping[str, str], name: str, default: int) -> int:
    value = _read_env(env, name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _read_bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _read_env(env, name)
    if not value:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean value.")


def _read_optional_path(env: Mapping[str, str], name: str) -> Path | None:
    value = _read_env(env, name)
    if not value:
        return None
    return Path(value).expanduser()
