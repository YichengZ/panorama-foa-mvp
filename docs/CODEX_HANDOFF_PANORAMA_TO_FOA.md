# Historical Handoff Note

This file previously contained the initial multi-agent implementation prompt
for creating `panorama_foa_mvp/` inside a SonoWorld checkout.

It is no longer the current server handoff document.

Use these current documents instead:

- `docs/SERVER_HANDOFF_MMAUDIO.md`
- `docs/MMAUDIO_LOCAL_BACKEND.md`
- `docs/IMMUTABLE_SCOPE.md`
- `panorama_foa_mvp/README.md`

Current canonical repository:

```text
https://github.com/YichengZ/panorama-foa-mvp
```

Current production audio direction:

```text
local MMAudio provider via --audio-provider mmaudio
```

Server validation must not call OpenAI or ElevenLabs unless explicitly
requested later.
