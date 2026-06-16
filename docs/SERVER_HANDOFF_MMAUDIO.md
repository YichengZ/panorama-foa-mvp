# Server Handoff: Panorama FOA Local MMAudio

## Branch

```bash
git clone https://github.com/YichengZ/panorama-foa-mvp.git
cd panorama-foa-mvp
git checkout main
git rev-parse HEAD
```

Expected handoff baseline: the latest pushed `main` commit from the personal
repo. If this document was updated, trust `main` over any older commit SHA in
chat history.

## Local Scope

The server should run only the `panorama_foa_mvp/` subproject.

Do not call OpenAI, ElevenLabs, or any paid API during server validation.

Do not enable SonoWorld's SAM3, Marble, segmentation, depth, point-cloud,
3DGS, player, frontend, HRTF, or true 6DoF stages.

## Environment

Use Python 3.11+ or 3.12, ffmpeg, CUDA/PyTorch compatible with the NVIDIA
driver, and the MMAudio dependencies required by SonoWorld's
`sonoworld.models.audio_diffusion.mmaudio.MMAudioDiffusion` wrapper.

Collect and report:

```bash
nvidia-smi
python --version
nvcc --version || true
df -h
free -h
```

Configure:

```bash
cd panorama_foa_mvp
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
cp .env.example .env
```

Recommended initial `.env` values:

```bash
MMAUDIO_MODEL_VARIANT=large_44k_v2
MMAUDIO_MODEL_PATH=.cache/weights/mmaudio
MMAUDIO_DEVICE=cuda:0
MMAUDIO_STEPS=25
MMAUDIO_GUIDANCE_SCALE=7.5
MMAUDIO_FULL_PRECISION=false
MMAUDIO_INFERENCE_MODE=euler
MMAUDIO_SONOWORLD_ROOT=
```

If `panorama_foa_mvp` is run outside the repository root, set
`MMAUDIO_SONOWORLD_ROOT` to the SonoWorld checkout path.

## Verification

Run the offline suite first:

```bash
cd panorama_foa_mvp
python -m pip install -e ".[test]"
pytest -q
python -m panorama_foa.cli generate \
  --panorama tests/fixtures/panorama_2x1.jpg \
  --output /tmp/panorama_foa_mock \
  --planner manual \
  --plan tests/fixtures/manual_plan.json \
  --audio-provider mock \
  --duration 1 \
  --force
```

Then run the real local MMAudio smoke test:

```bash
python -m panorama_foa.cli generate \
  --panorama assets/panoramas/starship_captain_private_quarters.png \
  --output /tmp/panorama_foa_starship_mmaudio \
  --planner manual \
  --plan examples/starship_captain_private_quarters_plan.json \
  --audio-provider mmaudio \
  --duration 15 \
  --yaw-offset 0 \
  --force
```

Expected files:

- `/tmp/panorama_foa_starship_mmaudio/scene_foa.wav`
- `/tmp/panorama_foa_starship_mmaudio/scene_foa.metadata.json`
- `/tmp/panorama_foa_starship_mmaudio/scene_plan.json`
- `/tmp/panorama_foa_starship_mmaudio/raw_audio/`
- `/tmp/panorama_foa_starship_mmaudio/stems/`
- `/tmp/panorama_foa_starship_mmaudio/panorama_analysis.jpg`

The final WAV must remain 4-channel, 48 kHz, `PCM_24`, AmbiX / ACN / SN3D,
with channel labels `[W,Y,Z,X]`.

Independent WAV/metadata readback:

```bash
python - <<'PY'
import json
import soundfile as sf
from pathlib import Path

out = Path("/tmp/panorama_foa_starship_mmaudio")
info = sf.info(out / "scene_foa.wav")
meta = json.loads((out / "scene_foa.metadata.json").read_text())

result = {
    "samplerate": info.samplerate,
    "channels": info.channels,
    "frames": info.frames,
    "subtype": info.subtype,
    "convention": meta["ambisonics"]["convention"],
    "channel_ordering": meta["ambisonics"]["channel_ordering"],
    "normalization": meta["ambisonics"]["normalization"],
    "channel_labels": meta["ambisonics"]["channel_labels"],
}
print(result)

assert result["samplerate"] == 48000
assert result["channels"] == 4
assert result["frames"] == 720000
assert result["subtype"] == "PCM_24"
assert result["convention"] == "AmbiX"
assert result["channel_ordering"] == "ACN"
assert result["normalization"] == "SN3D"
assert result["channel_labels"] == ["W", "Y", "Z", "X"]
PY
```

Final report must include the commit SHA, GPU/CUDA/Python environment, install
steps, `pytest -q` result, mock E2E result, real MMAudio result, output path,
WAV readback, and a clear statement that no paid API was called.
