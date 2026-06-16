# Examples

`manual_plan.example.json` shows the minimum scene plan shape accepted by the offline renderer.

Run with:

```bash
python -m panorama_foa.cli render \
  --plan examples/manual_plan.example.json \
  --output /tmp/panorama_foa_example \
  --audio-provider mock
```
