# MMAudio Local Backend

## Decision

Use local MMAudio inference as the preferred real text-to-audio backend for
`panorama_foa_mvp`. ElevenLabs remains available as an optional API fallback,
but the target production path is local model inference on the user's GPU
server.

## Available Server

The reported server class is 8x NVIDIA RTX 4090. This is sufficient for
MMAudio inference.

Recommended initial use:

- Start with one GPU and one generated stem at a time.
- Use 15-second stems.
- Keep `num_candidates=1` for the MVP.
- After quality is validated, parallelize independent stems across GPUs.

## Resource Baseline

Minimum practical inference host:

```text
GPU: NVIDIA GPU with 12GB+ VRAM
CPU: 8+ cores
RAM: 32GB+
Disk: 60GB+ free
OS: Ubuntu 22.04/24.04
Python: 3.12
CUDA: CUDA 12.x compatible driver and PyTorch build
```

Recommended host:

```text
GPU: RTX 4090 24GB or better
CPU: 16+ cores
RAM: 64GB+
Disk: 100GB+ NVMe free
OS: Ubuntu 22.04 LTS
Python: 3.12
```

For an 8x4090 host, memory is not the limiting factor for one 15-second stem.
The main engineering concern is environment correctness and scheduling.

## Model And Dependency Notes

The local backend for `panorama_foa_mvp` uses the project-local
`panorama_foa.audio.backends.mmaudio_diffusion.MMAudioDiffusion` wrapper. The
wrapper imports the external `mmaudio` package only when the real provider is
instantiated, so normal tests remain offline and GPU-free.

Expected MMAudio settings for this MVP:

```text
model_variant: large_44k_v2
seconds_total: 15.0
steps: 25
guidance_scale: 7.5
num_candidates: 1
sample_rate: 48000 after post-processing
```

Prefer reduced precision for inference if the upstream MMAudio package supports
it reliably on the server. Keep full precision as a fallback.

## Implemented Integration

`panorama_foa_mvp` exposes the local backend as:

```text
--audio-provider mmaudio
```

The provider:

- Implement `TextToAudioProvider.generate(...)`.
- Generate one mono-compatible raw stem per requested prompt.
- Avoid network calls during tests.
- Reuse the existing post-processing path: fold to mono, resample to 48 kHz,
  remove DC, fit exact duration, check silence, apply gain.
- Preserve the existing FOA encoder, mixer, WAV exporter, and metadata.
- Be covered by tests using a fake local MMAudio adapter.

Configuration is read from `.env` or environment variables:

```bash
MMAUDIO_MODEL_VARIANT=large_44k_v2
MMAUDIO_MODEL_PATH=.cache/weights/mmaudio
MMAUDIO_DEVICE=cuda:0
MMAUDIO_STEPS=25
MMAUDIO_GUIDANCE_SCALE=7.5
MMAUDIO_FULL_PRECISION=false
MMAUDIO_INFERENCE_MODE=euler
```

## Explicitly Not Required

Do not enable these out-of-scope systems for this MVP:

- SAM3 or any segmentation.
- Marble or any visual scene generation.
- Depth, point clouds, 3DGS, or 3D source coordinates.
- HRTF, player, frontend, WebXR, or true 6DoF.
- Any external spatialization stage outside this FOA encoder/mixer.

## Deployment Check Commands

Before running the real provider on the server, collect:

```bash
nvidia-smi
python --version
nvcc --version || true
df -h
free -h
```

Then install and smoke-test MMAudio independently before wiring it into the
FOA pipeline. After environment setup, run:

```bash
cd panorama_foa_mvp
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

Expected output files include `scene_foa.wav`, `scene_foa.metadata.json`,
`scene_plan.json`, `raw_audio/`, `stems/`, and `panorama_analysis.jpg`.
