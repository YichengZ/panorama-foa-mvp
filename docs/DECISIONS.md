# Decisions

## 2026-06-16

- Canonical repository: `https://github.com/YichengZ/panorama-foa-mvp`.
- Canonical branch: `main`.
- SonoWorld reference repository: `https://github.com/HuMathe/sonoworld.git`.
- Historical SonoWorld reference commit: `6a886ed734c03e34f09428a4b3d676ad726296a1`.
- The MVP is implemented as a self-contained `panorama_foa_mvp/` subproject.
- Upstream SonoWorld stages are reference-only and must not become runtime dependencies.
- FOA convention is fixed as AmbiX, ACN, SN3D, channel labels `[W,Y,Z,X]`.
- Tests use mock/manual providers by default and must not spend OpenAI or ElevenLabs credits.
- Verification Python: `/Users/zhuyicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` (`Python 3.12.13`).
- Verification ffmpeg: `/opt/homebrew/bin/ffmpeg` (`ffmpeg 8.1.1`).
- Homebrew `python@3.11` installed but its `ensurepip` postinstall fails on this host due to a `pyexpat`/system `libexpat` symbol mismatch, so the Codex bundled Python 3.12 runtime is used for project tests.

## 2026-06-16: Local Audio Backend Direction

- Target deployment for real sound generation is a local MMAudio backend, not ElevenLabs.
- Available server class reported by the user: 8x NVIDIA RTX 4090. This is more than sufficient for MMAudio inference; a single 4090 should be enough for one 15-second stem, and the 8-GPU server can later parallelize stems or jobs.
- ElevenLabs remains an implemented optional provider and useful API fallback, but it should not be treated as the preferred production path when the local server is available.
- `MMAudioTextToAudioProvider` is implemented for `panorama_foa_mvp` and exposed as `--audio-provider mmaudio`, while preserving the existing provider interface and FOA/mixer/export contracts.
- Integrating MMAudio into `panorama_foa_mvp` must not enable SAM3, Marble, depth, point-cloud, segmentation, player, frontend, or true 6DoF stages.
