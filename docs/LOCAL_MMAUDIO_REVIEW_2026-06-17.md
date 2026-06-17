# Local MMAudio Review - 2026-06-17

## Current State

The server-side local MMAudio MVP path is validated on commit
`922eb6650b2ef175abbd3dbe4b402d730f63ab6d`.

Validated path:

```text
panorama image
-> manual scene plan
-> local MMAudio text-to-audio
-> raw mono audio
-> stem processing with loop post-roll crossfade
-> FOA encoding and mix
-> AmbiX scene_foa.wav
-> scene_foa_loopcheck.wav for loop audition
```

The run did not use OpenAI, ElevenLabs, or paid APIs. The planner was manual and
the audio provider was local MMAudio.

## Kept Review Outputs

Review outputs are kept under the project validation folder, not in git:

```text
/mnt/afs_e/zhuyicheng/projects/panorama_foa_mmaudio_validation/outputs/panorama_foa_mock_loopqa
/mnt/afs_e/zhuyicheng/projects/panorama_foa_mmaudio_validation/outputs/panorama_foa_starship_mmaudio_loopqa_run_20260617_0700
```

Important real MMAudio files:

```text
scene_foa.wav              # formal 15 second FOA asset
scene_foa_loopcheck.wav    # 30 second two-repeat QA audition file
scene_foa.metadata.json    # AmbiX / ACN / SN3D / [W,Y,Z,X]
scene_foa.metrics.json     # peak, RMS, edge, and frame metrics
raw_audio/                 # MMAudio raw outputs, generated with loop post-roll
stems/                     # processed 15 second loopable stems
```

Readback from the real MMAudio loop QA run:

```text
scene_foa.wav:           48000 Hz, 4 channels, 720000 frames, PCM_24
scene_foa_loopcheck.wav: 48000 Hz, 4 channels, 1440000 frames, PCM_24
final RMS:               -30.00000006064086 dBFS
final peak:              -9.973030265133342 dBFS
metadata:                AmbiX / ACN / SN3D / [W,Y,Z,X]
```

## Cleanup Performed

Removed project-local redundant artifacts:

- Empty accidental git metadata backup from an earlier initialization mistake.
- Superseded patch and bundle files for commit `254f843`.
- Failed intermediate MMAudio smoke outputs.
- Old duplicate mock and pre-loop-QA output runs.
- Python test caches and egg-info generated during local testing.

Preserved:

- `model_cache/` because it contains required local MMAudio, CLIP, and BigVGAN
  weights.
- `third_party/MMAudio/` because the editable MMAudio package source is required.
- `panorama-foa-mvp/` because its GPU venv is still the validated local MMAudio
  runtime. This should be removed only after a GPU-capable venv is recreated
  under `panorama-foa-mvp-git/`.
- `panorama-foa-mvp.tar.gz` as provenance for the original target commit
  bootstrap.

## Engineering Review

The current design is usable for MVP validation, but the next productization
step should make loop/loudness behavior explicit instead of hard-coded.

Recommended next changes:

- Add a small `AudioRenderConfig` dataclass for loop post-roll seconds,
  loopcheck repeat count, target RMS dBFS, and peak ceiling dBFS.
- Move metrics generation into a dedicated audio QA module so pipeline orchestration
  stays thinner.
- Include render config and source generation durations in `scene_foa.metrics.json`
  for reproducibility.
- Create a single GPU setup script for the official git checkout and retire the
  old tarball checkout venv.
- Add a smoke command script that always clears paid API keys, clears proxies, sets
  local model cache paths, and writes to a timestamped output directory.
- Keep final asset and QA audition file separate: `scene_foa.wav` remains the
  deliverable, while `scene_foa_loopcheck.wav` is for human loop review.

## Verification Commands

Offline regression:

```bash
cd /mnt/afs_e/zhuyicheng/projects/panorama_foa_mmaudio_validation/panorama-foa-mvp-git/panorama_foa_mvp
PYTHONPATH=src ../.venv/bin/python -m pytest -q
```

Mock loop QA:

```bash
cd /mnt/afs_e/zhuyicheng/projects/panorama_foa_mmaudio_validation/panorama-foa-mvp-git/panorama_foa_mvp
PYTHONPATH=src ../.venv/bin/python -m panorama_foa.cli generate \
  --panorama tests/fixtures/panorama_2x1.jpg \
  --output /mnt/afs_e/zhuyicheng/projects/panorama_foa_mmaudio_validation/outputs/panorama_foa_mock_loopqa \
  --planner manual \
  --plan tests/fixtures/manual_plan.json \
  --audio-provider mock \
  --duration 1 \
  --force
```

Real local MMAudio validation should use the project-local model cache and clear
paid API keys. Keep the output in a timestamped folder under the validation
project's `outputs/` directory.
