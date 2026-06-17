# Panorama FOA MVP

This repository is the active development home for the Panorama to first-order
Ambisonics MVP:

```text
2:1 equirectangular panorama
-> static scene plan
-> mono-compatible text-to-audio stems
-> deterministic FOA AmbiX encoding
-> 4-channel 48 kHz PCM_24 WAV, ACN/SN3D [W,Y,Z,X]
```

The canonical repository is:

```text
https://github.com/YichengZ/panorama-foa-mvp
```

SonoWorld was used only as historical reference material. The runtime code in this repository is self-contained under `panorama_foa_mvp/` and must not enable SAM3, Marble, segmentation, depth, point-cloud, 3DGS, player, frontend, HRTF, or true 6DoF stages.

## Quick Start

```bash
cd panorama_foa_mvp
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
pytest -q
```

Offline mock end-to-end:

```bash
python -m panorama_foa.cli generate \
  --panorama tests/fixtures/panorama_2x1.jpg \
  --output /tmp/panorama_foa_mock \
  --planner manual \
  --plan tests/fixtures/manual_plan.json \
  --audio-provider mock \
  --duration 1 \
  --force
```

Server-side local MMAudio smoke test:

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

## Handoff

For server setup and validation, read:

- `docs/SERVER_HANDOFF_MMAUDIO.md`
- `docs/MMAUDIO_LOCAL_BACKEND.md`
- `docs/IMMUTABLE_SCOPE.md`
- `panorama_foa_mvp/README.md`
