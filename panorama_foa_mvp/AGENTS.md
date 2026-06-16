# Agent Instructions

- Preserve ACN/SN3D channel order `[W,Y,Z,X]`.
- Never normalize FOA channels independently.
- Tests must use mock providers and must not spend API credits.
- Do not add SAM3, Marble, HunyuanWorld, 3DGS, depth, point clouds, HRTF, a player, a frontend, or true 6DoF without an explicit task.
- Keep API providers behind interfaces.
- Keep input/output schemas backward compatible within `schema_version` 1.x.
- Every change to coordinate conventions requires updating tests and metadata.
- Treat `docs/IMMUTABLE_SCOPE.md` and `docs/ACCEPTANCE_MATRIX.md` as release gates.

