# Acceptance Matrix

| ID | Mandatory | Acceptance Item | Evidence |
| --- | --- | --- | --- |
| A01 | yes | `panorama_foa_mvp/` installs independently with Python 3.11+. | `pyproject.toml`, install command |
| A02 | yes | Input panorama aspect ratio is checked as approximately 2:1. | unit test |
| A03 | yes | Manual plan can skip VLM. | mock E2E test |
| A04 | yes | OpenAI planner uses Responses API image input and Pydantic structured output. | code review, mocked test |
| A05 | yes | ElevenLabs provider generates one raw layer per source with retries and no key logging. | code review, mocked HTTP test |
| A06 | yes | Tests never call paid APIs and pass without API keys. | network guard test |
| A07 | yes | Mono stems are 48 kHz, exact duration, DC removed, and normalized before `gain_db`. | audio processing test |
| A08 | yes | Coordinate mapping returns front 0, left +90, right -90, up +90 elevation. | coordinate test |
| A09 | yes | FOA output uses AmbiX, ACN, SN3D channel order `[W,Y,Z,X]`. | FOA/export tests |
| A10 | yes | Global ambience is W-only. | FOA test |
| A11 | yes | Spread 1.0 collapses to W-only directional output. | FOA test |
| A12 | yes | Mixer applies only one final scalar gain across all channels. | mixer test |
| A13 | yes | `scene_foa.wav` is 4-channel, 48000 Hz, PCM_24, exact frame count. | WAV readback |
| A14 | yes | Metadata declares ambisonic, channel, coordinate, yaw, and listener assumptions. | metadata test |
| A15 | yes | Mock CLI end-to-end produces plan, raw audio, stems, FOA WAV, metadata, and analysis image. | CLI E2E |
| A16 | yes | No Marble, SAM3, segmentation, 3DGS, depth, point clouds, HRTF, player, frontend, or true 6DoF. | scope guard |
| A17 | yes | `pytest -q` passes. | final command |
| A18 | yes | Local MMAudio provider config and CLI path are implemented without requiring network, GPU, or model weights in tests. | fake adapter tests |
| A19 | yes | Final `goal_guardian`, `acceptance_tester`, `independent_reviewer`, and `/review` pass with no unresolved P0/P1. | release packets |
