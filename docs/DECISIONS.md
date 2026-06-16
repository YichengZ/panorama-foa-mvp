# Decisions

## 2026-06-16

- Base repository: `https://github.com/HuMathe/sonoworld.git`.
- Base commit: `6a886ed734c03e34f09428a4b3d676ad726296a1`.
- Integration branch: `feat/panorama-to-foa-mvp`.
- The MVP is implemented as a self-contained `panorama_foa_mvp/` subproject.
- Upstream SonoWorld stages are reference-only and must not become runtime dependencies.
- FOA convention is fixed as AmbiX, ACN, SN3D, channel labels `[W,Y,Z,X]`.
- Tests use mock/manual providers by default and must not spend OpenAI or ElevenLabs credits.
- Verification Python: `/Users/zhuyicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` (`Python 3.12.13`).
- Verification ffmpeg: `/opt/homebrew/bin/ffmpeg` (`ffmpeg 8.1.1`).
- Homebrew `python@3.11` installed but its `ensurepip` postinstall fails on this host due to a `pyexpat`/system `libexpat` symbol mismatch, so the Codex bundled Python 3.12 runtime is used for project tests.
