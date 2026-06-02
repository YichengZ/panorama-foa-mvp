# SonoWorld: From One Image to a 3D Audio-Visual Scene (CVPR 2026)

> Official implementation of the CVPR 2026 paper _"SonoWorld: From One Image to a 3D Audio-Visual Scene."_
> 

**TL;DR** Given a single input image, SonoWorld generates a 3D audio-visual scene with spatialized sound and scene-level assets.

Dataset and full setup instructions are coming soon.


## Quick Start
After installing the required dependencies and preparing any model credentials/checkpoints needed by the selected stages, run:

```bash
python generate.py \
    --scene_root outputs/example_scene \
    --config configs/default.yaml \
    --input_image test-inputs/fall.jpg
```

Use `--resume` to continue a partially completed scene and `--force` to rerun completed stages.

## News and Planned TODOs

- [x] `06.02.2026` Released generation code
- [ ] Environment setup instructions
- [ ] SonoScene360 dataset
- [ ] Inference code, evaluation tools, interactive viewer, and additional examples
- [ ] Support for generation with open-source backbones, specifically HunyuanWorld 1.0 and LLaVA

## Installation
Coming soon.

## Citation
If you find our work useful, please cite:
```
@article{jin2026sonoworld,
    title={SonoWorld: From One Image to a 3D Audio-Visual Scene},
    author={Jin, Derong and Chen, Xiyi and Lin, Ming C. and Gao, Ruohan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```

## Licence
This project is released under the MIT Licence. See [LICENCE](LICENCE).

Third-party code included in this repository may be subject to its own license
terms, as noted in the corresponding source files.
