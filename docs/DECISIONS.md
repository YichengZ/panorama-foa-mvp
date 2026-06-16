# Decisions

## 2026-06-16

- Base repository: `https://github.com/HuMathe/sonoworld.git`.
- Base commit: `6a886ed734c03e34f09428a4b3d676ad726296a1`.
- Integration branch: `feat/panorama-to-foa-mvp`.
- The MVP is implemented as a self-contained `panorama_foa_mvp/` subproject.
- Upstream SonoWorld stages are reference-only and must not become runtime dependencies.
- FOA convention is fixed as AmbiX, ACN, SN3D, channel labels `[W,Y,Z,X]`.
- Tests use mock/manual providers by default and must not spend OpenAI or ElevenLabs credits.
- Current host has Homebrew prefixes for Python 3.11 and ffmpeg, but `python3` and `ffmpeg` are not both on PATH by default.

