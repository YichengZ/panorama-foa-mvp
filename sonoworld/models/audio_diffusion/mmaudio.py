from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sonoworld.models.audio_diffusion import (
    AudioDiffusionConditioning,
    AudioDiffusionConfig,
    AudioDiffusionModel,
)


@dataclass
class MMAudioConfig(AudioDiffusionConfig):
    spectrogram_frame_rate: int = 0
    sequence_cfg: Any = None
    sampler_type: str = "euler"


@dataclass
class MMAudioConditioning(AudioDiffusionConditioning):
    conditioning: Any = None
    neg_conditioning: Any = None
    class_labels: Any = None


class MMAudioDiffusion(AudioDiffusionModel):
    """Release wrapper around the upstream `mmaudio` package.

    The external package and torch are imported only when this class is
    instantiated. Importing `sonoworld.models.audio_diffusion.mmaudio` itself is
    therefore safe in environments that do not have MMAudio installed.
    """

    supports_t2a = True
    supports_v2a = True

    def __init__(
        self,
        model_name: str = "mmaudio",
        variant: str = "large_44k_v2",
        steps: int = 25,
        guidance_scale: float = 7.5,
        seconds_total: float = 8.0,
        full_precision: bool = True,
        inference_mode: str = "euler",
        model_path: str | Path = ".cache/weights/mmaudio",
        device: Optional[str] = None,
    ) -> None:
        if model_name != "mmaudio":
            raise ValueError(f"MMAudioDiffusion only supports model_name='mmaudio', got {model_name!r}.")

        import torch
        from mmaudio.eval_utils import all_model_cfg, load_video
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        self.load_video = load_video

        self.model_name = model_name
        self.variant = variant
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.seconds_total = float(seconds_total)
        self.inference_mode = inference_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 if full_precision else torch.bfloat16

        if variant not in all_model_cfg:
            available = ", ".join(sorted(all_model_cfg.keys()))
            raise ValueError(f"Unknown MMAudio variant {variant!r}. Available variants: {available}")

        model_cfg = copy.copy(all_model_cfg[variant])
        sequence_cfg = copy.copy(model_cfg.seq_cfg)

        if model_path is not None:
            model_root = Path(model_path)
            model_cfg.model_path = model_root / model_cfg.model_path.name
            model_cfg.vae_path = model_root / model_cfg.vae_path.name
            model_cfg.synchformer_ckpt = model_root / model_cfg.synchformer_ckpt.name
            if model_cfg.bigvgan_16k_path:
                model_cfg.bigvgan_16k_path = model_root / model_cfg.bigvgan_16k_path.name

        model_cfg.download_if_needed()
        self.sequence_cfg = sequence_cfg
        self.sequence_cfg.duration = self.seconds_total

        self.model = get_my_mmaudio(model_cfg.model_name).to(self.dtype)
        self.model.load_weights(
            self._torch_load_weights(model_cfg.model_path),
        )
        self.model.update_seq_lengths(
            self.sequence_cfg.latent_seq_len,
            self.sequence_cfg.clip_seq_len,
            self.sequence_cfg.sync_seq_len,
        )
        self.model.to(self.device).eval()

        self.feature_utils = FeaturesUtils(
            tod_vae_ckpt=model_cfg.vae_path,
            synchformer_ckpt=model_cfg.synchformer_ckpt,
            enable_conditions=True,
            mode=model_cfg.mode,
            bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
            need_vae_encoder=True,
        ).eval()
        self.feature_utils.to(self.dtype).to(self.device)

        self.flow_matching = FlowMatching(
            min_sigma=0.0,
            inference_mode=inference_mode,
            num_steps=self.steps,
        )

    def get_config(self) -> MMAudioConfig:
        return MMAudioConfig(
            sample_rate=int(self.sequence_cfg.sampling_rate),
            steps=self.steps,
            guidance_scale=self.guidance_scale,
            seconds_start=0.0,
            seconds_total=self.seconds_total,
            model_name=self.model_name,
            variant=self.variant,
            spectrogram_frame_rate=int(self.sequence_cfg.spectrogram_frame_rate),
            sequence_cfg=self.sequence_cfg,
            sampler_type=self.inference_mode,
        )

    def get_conditioning(
        self,
        text_prompt: Optional[str] = None,
        video_path: Optional[str | Path] = None,
    ) -> MMAudioConditioning:
        batch_size = 1
        clip_batch_size_multiplier = 40
        sync_batch_size_multiplier = 40

        if video_path is not None:
            video_info = self.load_video(str(video_path), self.seconds_total)
            clip_frames = video_info.clip_frames.unsqueeze(0)
            sync_frames = video_info.sync_frames.unsqueeze(0)

            clip_frames = clip_frames.to(self.device, self.dtype, non_blocking=True)
            clip_features = self.feature_utils.encode_video_with_clip(
                clip_frames,
                batch_size=clip_batch_size_multiplier * batch_size,
            )

            sync_frames = sync_frames.to(self.device, self.dtype, non_blocking=True)
            sync_features = self.feature_utils.encode_video_with_sync(
                sync_frames,
                batch_size=sync_batch_size_multiplier * batch_size,
            )
        else:
            clip_features = self.model.get_empty_clip_sequence(batch_size)
            sync_features = self.model.get_empty_sync_sequence(batch_size)

        if text_prompt is not None:
            text_features = self.feature_utils.encode_text([text_prompt])
        else:
            text_features = self.model.get_empty_string_sequence(batch_size)

        conditioning = self.model.preprocess_conditions(
            clip_features.to(self.dtype),
            sync_features.to(self.dtype),
            text_features.to(self.dtype),
        )
        neg_conditioning = self.model.get_empty_conditions(batch_size)
        return MMAudioConditioning(
            conditioning=conditioning,
            neg_conditioning=neg_conditioning,
        )

    def generate_t2a(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        condition: Optional[MMAudioConditioning] = None,
        config: Optional[MMAudioConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return self.generate_audio(
            text_prompt=prompt,
            condition=condition,
            config=config,
        )

    def generate_v2a(
        self,
        video_path: str | Path,
        *,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        condition: Optional[MMAudioConditioning] = None,
        config: Optional[MMAudioConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return self.generate_audio(
            text_prompt=prompt,
            video_path=video_path,
            condition=condition,
            config=config,
        )

    def generate_audio(
        self,
        *,
        text_prompt: Optional[str] = None,
        video_path: Optional[str | Path] = None,
        condition: Optional[MMAudioConditioning] = None,
        config: Optional[MMAudioConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if config is None:
            config = self.get_config()
        if condition is None:
            condition = self.get_conditioning(text_prompt=text_prompt, video_path=video_path)

        import torch

        with torch.no_grad():
            x0 = torch.randn(
                1,
                self.model.latent_seq_len,
                self.model.latent_dim,
                device=self.device,
                dtype=self.dtype,
            )
            cfg_ode_wrapper = lambda t, x: self.model.ode_wrapper(
                t,
                x,
                condition.conditioning,
                condition.neg_conditioning,
                config.guidance_scale,
            )
            x1 = self.flow_matching.to_data(cfg_ode_wrapper, x0)
            x1 = self.model.unnormalize(x1)
            spec = self.feature_utils.decode(x1)
            audio = self.feature_utils.vocode(spec)

        return audio

    def get_flow(self, z_t: Any, t: Any, condition: MMAudioConditioning, config: MMAudioConfig) -> Any:
        return _ode_wrapper(
            self.model,
            t,
            z_t,
            condition.conditioning,
            condition.neg_conditioning,
            config.guidance_scale,
        )

    def get_v(self, z_t: Any, t: Any, condition: MMAudioConditioning, config: MMAudioConfig) -> Any:
        raise ValueError("MMAudio does not support v prediction; use get_flow instead.")

    def get_x0(self, z_t: Any, t: Any, condition: MMAudioConditioning, config: MMAudioConfig) -> Any:
        flow = self.get_flow(z_t, t, condition, config)
        return z_t - flow * t

    def get_epsilon(
        self,
        z_t: Any,
        t: Any,
        condition: MMAudioConditioning,
        config: MMAudioConfig,
    ) -> Any:
        raise ValueError("MMAudio does not support epsilon prediction; use get_flow instead.")

    def encode_audio(self, audio: Any, return_mel: bool = False) -> Any:
        audio = audio[:, 0]
        mel = self.feature_utils.mel_converter(audio)
        latents_vae = self.feature_utils.tod.vae.encode(mel).sample().transpose(1, 2).to(self.dtype)
        latents = self.model.normalize(latents_vae)
        if return_mel:
            return latents, mel
        return latents

    def decode_audio(self, latents: Any, return_mel: bool = False) -> Any:
        latents = latents.to(self.dtype)
        latents_vae = self.model.unnormalize(latents)
        mel = self.feature_utils.tod.vae.decode(
            latents_vae.transpose(1, 2).to(self.feature_utils.dtype)
        )
        audio = self.feature_utils.vocode(mel)
        if return_mel:
            return audio, mel
        return audio

    def multi_step_denoising_euler(
        self,
        noisy_latents: Any,
        start_t: Any,
        condition: MMAudioConditioning,
        steps: int,
        guidance_scale: Optional[float] = None,
    ) -> Any:
        config = self.get_config()
        guidance_scale = guidance_scale if guidance_scale is not None else config.guidance_scale
        fn = lambda t, x: _ode_wrapper(
            self.model,
            t,
            x,
            condition.conditioning,
            condition.neg_conditioning,
            guidance_scale,
        )

        import torch

        scale = torch.linspace(0.0, 1.0, steps=steps + 1, device=noisy_latents.device)
        timesteps = start_t.unsqueeze(1) + scale.unsqueeze(0) * (1.0 - start_t.unsqueeze(1))
        timesteps = timesteps.to(self.dtype)
        x = noisy_latents
        for ti in range(steps):
            t = timesteps[:, ti]
            next_t = timesteps[:, ti + 1]
            flow = fn(t, x)
            dt = next_t - t
            x = x + dt * flow
        return x

    def multi_step_denoising_eular(self, *args: Any, **kwargs: Any) -> Any:
        return self.multi_step_denoising_euler(*args, **kwargs)

    def eval(self) -> "MMAudioDiffusion":
        self.model.eval()
        self.feature_utils.eval()
        return self

    def to(self, device: str) -> "MMAudioDiffusion":
        self.device = str(device)
        self.model.to(self.device)
        self.feature_utils.to(self.device)
        return self

    def _torch_load_weights(self, path: Path) -> Any:
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

def _ode_wrapper(
    model: Any,
    t: Any,
    latent: Any,
    conditions: Any,
    empty_conditions: Any,
    cfg_strength: float,
) -> Any:
    if cfg_strength < 1.0:
        return model.predict_flow(latent, t, conditions)

    return (
        cfg_strength * model.predict_flow(latent, t, conditions)
        + (1.0 - cfg_strength) * model.predict_flow(latent, t, empty_conditions)
    )
