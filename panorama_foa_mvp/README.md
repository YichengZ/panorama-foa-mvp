# Panorama FOA MVP

This subproject converts a 2:1 equirectangular panorama into a static first-order Ambisonics WAV.

It is an MVP for a fixed listening point:

- Input: one complete equirectangular panorama.
- Planning: VLM or manual JSON scene plan.
- Audio generation: isolated mono-compatible stems.
- Output: 4-channel, 48 kHz FOA WAV using AmbiX / ACN / SN3D channel labels `[W,Y,Z,X]`.

It is not true 6DoF audio. Listener translation does not change audio. This project does not implement a player, binaural preview, physical acoustics, distance estimation, or pixel-level source localization.

## Backend Direction

The preferred production audio backend is local MMAudio inference. ElevenLabs is implemented as an optional API fallback, but it is not the target backend when a local GPU server is available.

The current implementation supports:

- `--audio-provider mock` for offline tests and deterministic development.
- `--audio-provider elevenlabs` for optional paid API generation.

Planned next backend:

- `--audio-provider mmaudio` using a local MMAudio installation to generate mono-compatible stems that are then post-processed and encoded by this project into FOA.

MMAudio integration must stay inside this subproject's provider interface. It must not enable or depend on SonoWorld's SAM3, Marble, segmentation, depth, point-cloud, player, frontend, or true 6DoF stages.

## Install

Use Python 3.11 or newer and install `ffmpeg`.

```bash
cd panorama_foa_mvp
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

## Mock End-to-End

```bash
python -m panorama_foa.cli generate \
  --panorama tests/fixtures/panorama_2x1.jpg \
  --output /tmp/panorama_foa_test \
  --planner manual \
  --plan tests/fixtures/manual_plan.json \
  --audio-provider mock
```

## Local MMAudio Deployment Target

An 8x NVIDIA RTX 4090 server is sufficient for MMAudio inference. A single 4090 should be enough for one 15-second stem; the full 8-GPU server can later be used to parallelize independent stems or jobs.

Recommended server baseline:

```text
GPU: NVIDIA RTX 4090 24GB, one or more
CPU: 16+ cores
RAM: 64GB+
Disk: 100GB+ free NVMe for environments, model weights, and outputs
OS: Ubuntu 22.04 LTS or newer
Python: 3.12
CUDA/PyTorch: match the installed NVIDIA driver; SonoWorld was tested with CUDA 12.4.1
```

For the MVP scene size, assume at most six generated stems: one global ambience plus up to five regional sources. Generation can be sequential on one GPU first, then parallelized after quality and stability are confirmed.

## Optional ElevenLabs API Commands

Configure `.env` from `.env.example` and use:

```bash
python -m panorama_foa.cli generate \
  --panorama ../test-inputs/panorama.jpg \
  --output ../outputs/park_001 \
  --duration 16 \
  --audio-provider elevenlabs \
  --yaw-offset 0
```

Render from an existing plan without VLM:

```bash
python -m panorama_foa.cli render \
  --plan ../outputs/park_001/scene_plan.json \
  --output ../outputs/park_001 \
  --audio-provider elevenlabs
```

## Outputs

Each run writes `scene_plan.json`, `scene_foa.wav`, `scene_foa.metadata.json`, `panorama_analysis.jpg`, `raw_audio/`, and `stems/`.

The WAV header alone does not fully describe Ambisonics semantics, so always keep the metadata JSON with the audio.

## Attribution

This subproject lives inside the SonoWorld repository and is inspired by SonoWorld's image understanding and layered audio-generation structure, but it is independent from the upstream CUDA, 3D, segmentation, rendering, and viewer stages.

## Tests

```bash
pytest -q
```

Tests use mock providers or mocked HTTP clients and must pass without OpenAI or ElevenLabs API keys.

## Known Limits

- Fixed listener position only.
- No distance, occlusion, reflection, depth, point cloud, segmentation, or physical room model.
- No player, Web UI, HRTF/HRIR, binaural preview, or realtime rendering.
- VLM source positions are coarse normalized panorama coordinates intended for an editable first pass.
