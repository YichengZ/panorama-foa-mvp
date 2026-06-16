# Immutable Scope: Panorama to FOA MVP

This project adds one independent subproject, `panorama_foa_mvp/`, to the SonoWorld repository.

## In Scope

- Validate a single 2:1 equirectangular panorama.
- Produce a VLM or manual `ScenePlan`.
- Generate isolated mono text-to-audio stems.
- Encode and mix deterministic first-order Ambisonics.
- Export a 4-channel 48 kHz WAV using AmbiX, ACN, SN3D, channel labels `[W,Y,Z,X]`.
- Provide metadata that states coordinate and ambisonic conventions.
- Provide mock providers and tests that run without paid API calls.

## Out of Scope

- Marble.
- HunyuanWorld.
- SAM3, SAM2, X-Decoder, segmentation, or segmentation voting.
- 3D Gaussian Splatting.
- Depth, point clouds, 3D source coordinates, occlusion, diffraction, room reflections, distance attenuation, or air absorption.
- HRTF, HRIR, binaural playback, players, Web UI, Three.js, WebXR, mobile sources, video sync, or true 6DoF.

Any implementation, dependency, README claim, test fixture, or code path that adds an out-of-scope item is a release-blocking failure.
