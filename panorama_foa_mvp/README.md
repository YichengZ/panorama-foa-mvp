# Panorama FOA MVP

This subproject converts a 2:1 equirectangular panorama into a static first-order Ambisonics WAV.

It is an MVP for a fixed listening point:

- Input: one complete equirectangular panorama.
- Planning: VLM or manual JSON scene plan.
- Audio generation: isolated mono-compatible stems.
- Output: 4-channel, 48 kHz FOA WAV using AmbiX / ACN / SN3D channel labels `[W,Y,Z,X]`.

It is not true 6DoF audio. Listener translation does not change audio. This project does not implement a player, binaural preview, physical acoustics, distance estimation, or pixel-level source localization.

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

## Real API Commands

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
