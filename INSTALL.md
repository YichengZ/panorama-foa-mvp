# Install

## MVP Environment

Use Python 3.11+ or 3.12.

```bash
git clone https://github.com/YichengZ/panorama-foa-mvp.git
cd panorama-foa-mvp/panorama_foa_mvp
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
pytest -q
```

`ffmpeg` should be available on `PATH` for non-WAV provider outputs.

## Local MMAudio Server

The real audio backend for server validation is local MMAudio:

```bash
cp .env.example .env
```

Recommended initial values:

```bash
MMAUDIO_MODEL_VARIANT=large_44k_v2
MMAUDIO_MODEL_PATH=.cache/weights/mmaudio
MMAUDIO_DEVICE=cuda:0
MMAUDIO_STEPS=25
MMAUDIO_GUIDANCE_SCALE=7.5
MMAUDIO_FULL_PRECISION=false
MMAUDIO_INFERENCE_MODE=euler
```

Install the MMAudio dependency stack required by the project-local `panorama_foa.audio.backends.mmaudio_diffusion.MMAudioDiffusion` wrapper on the GPU server. Do not install or enable SAM3, Marble, segmentation, depth, point-cloud, 3DGS, player, frontend, HRTF, or true 6DoF stages for this MVP.

Run:

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

