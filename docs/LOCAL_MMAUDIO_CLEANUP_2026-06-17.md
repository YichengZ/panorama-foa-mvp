# Local MMAudio Cleanup - 2026-06-17

## Workspace

The local validation workspace was rebuilt as:

```text
/mnt/afs_e/zhuyicheng/projects/panorama_foa_mvp_local_mmaudio/
  repo/
  models/
  third_party/MMAudio/
  review_outputs/
  provenance/
```

The old `panorama_foa_mmaudio_validation/` directory and the temporary
`legacy_gpu_checkout/` were removed after the cleaned repo successfully ran a
real local MMAudio smoke test.

## Repository Cleanup

- Removed the old root SonoWorld pipeline, stages, configs, and `test-inputs/fall.jpg`.
- Moved the minimal MMAudio diffusion wrapper into
  `panorama_foa.audio.backends.mmaudio_diffusion`.
- Removed `MMAUDIO_SONOWORLD_ROOT`; runtime no longer depends on a SonoWorld
  checkout.
- Added `AudioRenderConfig` and `audio_metrics` so loop QA and loudness settings
  are explicit.
- Kept OpenAI and ElevenLabs as optional providers only; server validation uses
  manual planning and local MMAudio.

## Verification

Offline regression:

```text
59 passed
```

Mock E2E readback:

```text
scene_foa.wav:           48000 Hz, 4 channels, 48000 frames, PCM_24
scene_foa_loopcheck.wav: 48000 Hz, 4 channels, 96000 frames, PCM_24
metadata:                AmbiX / ACN / SN3D / [W,Y,Z,X]
```

Real local MMAudio smoke:

```text
output:                  /mnt/afs_e/zhuyicheng/projects/panorama_foa_mvp_local_mmaudio/review_outputs/panorama_foa_starship_mmaudio_clean_run_20260617
scene_foa.wav:           48000 Hz, 4 channels, 720000 frames, PCM_24
scene_foa_loopcheck.wav: 48000 Hz, 4 channels, 1440000 frames, PCM_24
metadata:                AmbiX / ACN / SN3D / [W,Y,Z,X]
final RMS:               -30.000000210931578 dBFS
final peak:              -16.851570591671226 dBFS
```

The successful smoke used local Hugging Face cache only:

```bash
HF_HOME=/mnt/afs_e/zhuyicheng/projects/panorama_foa_mvp_local_mmaudio/models/huggingface
HUGGINGFACE_HUB_CACHE=/mnt/afs_e/zhuyicheng/projects/panorama_foa_mvp_local_mmaudio/models/huggingface/hub
HF_HUB_OFFLINE=1
```

OpenAI, ElevenLabs, and paid APIs were not called.

