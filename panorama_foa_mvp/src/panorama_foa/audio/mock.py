from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


class MockTextToAudioProvider:
    sample_rate = 48000
    raw_extension = ".wav"

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        loop: bool,
        output_path: Path,
    ) -> Path:
        _ = loop
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples = round(duration_seconds * self.sample_rate)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        frequency = 180.0 + (int.from_bytes(digest[:2], "big") % 600)
        t = np.arange(samples, dtype=np.float32) / self.sample_rate

        if "global" in prompt.lower() or "background" in prompt.lower() or "ambience" in prompt.lower():
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            signal = rng.normal(0.0, 0.08, size=samples).astype(np.float32)
            if samples > 8:
                kernel = np.ones(9, dtype=np.float32) / 9.0
                signal = np.convolve(signal, kernel, mode="same").astype(np.float32)
        else:
            signal = (0.45 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)

        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        if peak > 0.5:
            signal *= 0.5 / peak
        sf.write(output_path, signal, self.sample_rate, subtype="PCM_16")
        return output_path
