# Immutable Scope: Panorama to FOA MVP

This project is the independent `panorama_foa_mvp/` MVP in the personal
`YichengZ/panorama-foa-mvp` repository. SonoWorld is reference code only.

## In Scope

- Validate a single 2:1 equirectangular panorama.
- Produce a VLM or manual `ScenePlan`.
- Generate isolated mono text-to-audio stems.
- Generate real stems through the local `MMAudioTextToAudioProvider` when MMAudio is installed on a GPU server.
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
