# Implementation Plan: Panorama to FOA MVP

## Gate Status

Implementation remains blocked until an independent `goal_guardian` reviews this file and returns `GO`.

Pre-implementation packets:

- `repo_explorer`: PASS. Confirmed branch `feat/panorama-to-foa-mvp`, base `6a886ed734c03e34f09428a4b3d676ad726296a1`, clean worktree, independent subproject strategy, and upstream reference paths.
- `test_designer`: PASS. Produced acceptance test contract for coordinates, FOA, mixer, exporter, metadata, mock E2E, paid API blocking, and forbidden-scope guard.
- `goal_guardian`: BLOCKED by missing `IMPLEMENTATION_PLAN.md`. Required this plan, final reviewer configs, explicit file ownership, and GO before implementation.

## Worktree and Ownership Plan

Use isolated git worktrees for writing agents. If worktree setup fails, use the serialized fallback order exactly: `test_worker -> foa_worker -> pipeline_worker -> api_worker`.

Branches and owners:

- `agent/test-contract`: `test_worker`
  - Owns `panorama_foa_mvp/tests/**` and `panorama_foa_mvp/examples/**`.
  - Must not edit production code.
- `agent/foa-core`: `foa_worker`
  - Owns `src/panorama_foa/coordinates.py` and `src/panorama_foa/ambisonics/**`.
  - Must preserve AmbiX ACN/SN3D `[W,Y,Z,X]` and single scalar mix limiting.
- `agent/offline-pipeline`: `pipeline_worker`
  - Owns schemas, CLI, pipeline, manual planner, mock provider, decode, and processing files.
  - Must make manual + mock E2E work without network or paid APIs.
- `agent/api-providers`: `api_worker`
  - Owns config, image utilities, OpenAI VLM planner, ElevenLabs provider, prompt, and `.env.example`.
  - Must keep API providers behind interfaces and avoid key logging.

Shared scaffold is created by the main coordinator before worker code where necessary:

- `pyproject.toml`, `README.md`, package `__init__.py` files, `planner/base.py`, `audio/provider_base.py`.
- Workers may adjust imports in these shared files only after coordinator review and without changing another worker's behavior.

## Implementation Order

1. Environment precheck:
   - Ensure Python 3.11+ is available.
   - Ensure `ffmpeg` is available for decode paths.
   - Record exact tool paths in `docs/DECISIONS.md`.
2. `test_worker` creates and commits acceptance tests and fixtures first.
3. `foa_worker` implements coordinates, FOA encoder, global ambience, mixer, exporter, and metadata helpers.
4. `pipeline_worker` implements Pydantic schemas, manual planner, mock provider, audio post-processing, pipeline, and Typer CLI offline flow.
5. `api_worker` implements OpenAI Responses API VLM planner, ElevenLabs provider with retries, image validation/downscale, prompt, and env config.
6. Main coordinator integrates branches in order, running targeted tests after each merge.
7. Main coordinator completes docs, examples, and final validation.

## Acceptance Mapping

- A01: `pyproject.toml` with Python `>=3.11`; install and test command evidence.
- A02: image utility tests for 2:1 panorama validation.
- A03: manual planner and `render`/`generate --planner manual` tests.
- A04: mocked OpenAI planner test or code review evidence for Responses API plus Pydantic structured output.
- A05: mocked ElevenLabs HTTP tests for request shape, retry behavior, raw file saving, and no key logging.
- A06: network guard and no-key test environment.
- A07: audio processing tests for mono, 48 kHz, exact duration, DC removal, normalization, gain, and silence failure.
- A08: coordinate tests for front, left, right, back, up, down, and yaw wrap.
- A09: FOA/export tests for AmbiX ACN/SN3D `[W,Y,Z,X]`.
- A10: W-only global ambience test.
- A11: spread `1.0` W-only directional collapse test.
- A12: mixer test proving one final scalar across all channels and no in-place mutation.
- A13: soundfile readback test for 4 channels, 48000 Hz, exact frames, PCM_24.
- A14: metadata JSON test for ambisonics, coordinates, yaw, and listener model.
- A15: mock CLI E2E test for expected output tree.
- A16: scope guard scanning code/docs/deps for forbidden scope.
- A17: final `pytest -q`.
- A18: final independent `goal_guardian`, `acceptance_tester`, `independent_reviewer`, then Codex `/review`.

## Test Contract Details

Required tests:

- `test_coordinates.py`: normalized panorama mapping and yaw wrap to `[-180, 180)`.
- `test_foa_encoder.py`: front/left/right/back/up/full-spread math, `float32`, shape `(N, 4)`, channel order `[W,Y,Z,X]`.
- `test_mixer.py`: layer summing, final target peak not above `-1 dBFS`, one scalar gain, no per-channel normalization, no input mutation.
- `test_exporter.py`: WAV readback and metadata content.
- `test_audio_processing.py`: multichannel fold-down, resample or exact-rate handling, target length, DC removal, `-6 dBFS` reference peak before `gain_db`, near-silent source errors.
- `test_pipeline_mock.py`: manual plan + mock provider offline E2E, CLI duration override, low confidence filtering, max five regionals.
- `test_no_paid_api.py`: no real OpenAI/ElevenLabs requests in tests, no API keys required.
- `test_scope_guard.py`: forbidden dependency/content scan.

## Verification Commands

Run during integration and final gates:

```bash
cd panorama_foa_mvp
python -m pytest -q
python -m panorama_foa.cli generate \
  --panorama tests/fixtures/panorama_2x1.jpg \
  --output /tmp/panorama_foa_test \
  --planner manual \
  --plan tests/fixtures/manual_plan.json \
  --audio-provider mock
python - <<'PY'
from pathlib import Path
import json
import soundfile as sf
out = Path('/tmp/panorama_foa_test')
info = sf.info(out / 'scene_foa.wav')
meta = json.loads((out / 'scene_foa.metadata.json').read_text())
assert info.samplerate == 48000
assert info.channels == 4
assert info.frames == round(meta['duration_seconds'] * 48000)
assert meta['ambisonics']['channel_labels'] == ['W', 'Y', 'Z', 'X']
assert meta['ambisonics']['channel_ordering'] == 'ACN'
assert meta['ambisonics']['normalization'] == 'SN3D'
PY
```

Also run:

```bash
rg -n "Marble|SAM3|SAM2|X-Decoder|segmentation voting|3DGS|HRTF|HRIR|Three\\.js|WebXR|true 6DoF|torch|torchaudio|MMAudio|librosa" panorama_foa_mvp
```

This command should return no forbidden implementation/dependency/claim matches except explicit scope warnings in docs/tests.

## API Policy

- Unit and E2E tests must use mock providers or mocked HTTP clients.
- Real OpenAI or ElevenLabs calls are opt-in only and are not required for release.
- If API keys are absent, final report must say the API provider is implemented and tested with mocks, but no paid online call was executed.
- Never print API keys or include them in exception text.

## Final Release Gate

After integration, spawn `goal_guardian`, `acceptance_tester`, and `independent_reviewer` in parallel. Wait for all three before acting on results. Then run Codex `/review` on the integrated diff.

FAIL if any mandatory acceptance item fails, any unresolved P0/P1 exists, FOA coordinates/order are wrong, tests make real paid API calls, or any forbidden scope appears.
