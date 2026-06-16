---
name: panorama-foa-release-gate
description: Release gate for the Panorama to FOA MVP.
---

# Panorama FOA Release Gate

Use this gate before declaring completion.

## Mandatory Checks

- All mandatory rows in `docs/ACCEPTANCE_MATRIX.md` pass.
- `cd panorama_foa_mvp && pytest -q` passes.
- Mock CLI end-to-end generation passes without network or paid API calls.
- Independent WAV readback proves 4 channels, 48000 Hz, exact frame count, and 24-bit PCM WAV.
- `scene_foa.metadata.json` declares AmbiX, ACN, SN3D, and channel labels `[W,Y,Z,X]`.
- FOA coordinate and direction tests pass for front, left, right, back, up, and full spread.
- Final mix uses one global scalar when limiting peak; channels are never normalized independently.
- No unresolved P0/P1 findings remain.
- No Marble, SAM3, segmentation voting, HunyuanWorld, 3DGS, depth, point cloud, HRTF, player, frontend, or true 6DoF implementation appears.

## Required Verdict

Return `PASS` only when every mandatory check passes with evidence. Return `FAIL` for any missing evidence or failed mandatory item.

