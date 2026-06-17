# Implementation Status

This document records the current state of the Panorama FOA MVP. It is no
longer a pre-implementation gate.

## Current Repository

- Canonical repo: `https://github.com/YichengZ/panorama-foa-mvp`
- Main branch: `main`
- MVP subproject: `panorama_foa_mvp/`
- Historical reference codebase: SonoWorld. The local MMAudio wrapper is now vendored into `panorama_foa.audio.backends.mmaudio_diffusion`.

## Implemented

- 2:1 equirectangular panorama validation.
- Pydantic v2 scene plan schema.
- Manual planner and OpenAI VLM planner interface.
- Mock text-to-audio provider for offline tests.
- Local MMAudio text-to-audio provider behind `--audio-provider mmaudio`.
- Optional ElevenLabs provider kept as explicit fallback, not the server path.
- Mono stem post-processing: fold-down, resampling to 48 kHz, DC removal,
  exact duration fitting, silence rejection, and gain.
- FOA encoding with AmbiX / ACN / SN3D channel labels `[W,Y,Z,X]`.
- Mixer with one final scalar limiter, no per-channel normalization.
- 4-channel 48 kHz `PCM_24` WAV export and metadata.
- Starship cabin panorama asset and 15-second manual scene plan.
- Server handoff documentation.

## Required Server Work

The next agent should run server-side local MMAudio validation:

1. Clone `https://github.com/YichengZ/panorama-foa-mvp`.
2. Read `docs/SERVER_HANDOFF_MMAUDIO.md`.
3. Install Python dependencies and run `pytest -q`.
4. Run the mock end-to-end command.
5. Install MMAudio dependencies and model weights on the GPU server.
6. Run the real `--audio-provider mmaudio` smoke test for the starship cabin.
7. Independently read back the WAV and metadata.

## Scope Guard

Do not enable SAM3, Marble, segmentation, depth, point-cloud, 3DGS, player, frontend, HRTF, or true 6DoF stages.

Do not call OpenAI or ElevenLabs during server smoke testing unless explicitly
requested later. The server target is local MMAudio.
