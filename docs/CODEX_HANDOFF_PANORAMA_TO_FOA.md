# Codex 工程交接：全景图生成四声道 Ambisonics（FOA）MVP

> 文档日期：2026-06-16
> 参考仓库：`https://github.com/HuMathe/sonoworld`
> 目标：Codex clone SonoWorld 后，在不依赖 Marble、3DGS、深度、SAM3 的前提下，实现一套独立可运行的“全景图 → 四声道 Ambisonics WAV”工程。

---

## 0. 给 Codex 的执行指令

请先完整阅读本文件，再开始修改代码。

你的任务不是完整复现 SonoWorld，也不是补全其 3D/6DoF renderer。请在 clone 后的仓库中新增一个自包含子项目 `panorama_foa_mvp/`，保留上游原有代码，借鉴其 VLM 声音规划和分层生成思想，实现本文定义的最小产品。

执行原则：

1. 不等待额外澄清，按本文默认值执行。
2. 不修改或删除上游 SonoWorld 原有阶段。
3. 不引入 Marble、HunyuanWorld、3DGS、深度图、点云、SAM3、分割投票、HRTF 播放器或 Web UI。
4. 默认使用 VLM 直接输出声音候选及其全景图归一化坐标。
5. 默认使用 ElevenLabs Sound Effects API 生成单声道兼容的声音素材；同时必须实现不消耗 API 的 mock provider，用于测试。
6. Ambisonics 必须由本项目自行编码和混合，不要求 Text-to-Audio 模型直接生成四声道音频。
7. 所有自动化测试通过后才算完成。
8. 若复制上游源码，保留 MIT 许可和必要 attribution；不要复制不需要的第三方模型代码。

建议先执行：

```bash
git clone https://github.com/HuMathe/sonoworld.git
cd sonoworld
git checkout -b feat/panorama-to-foa-mvp
```

然后在仓库根目录新增：

```text
panorama_foa_mvp/
```

---

## 1. 核心需求

### 1.1 输入

一张完整的 2:1 等距柱状投影全景图：

```text
JPG 或 PNG
建议分辨率：4096 × 2048
允许其他 2:1 分辨率
```

可选参数：

- 用户补充场景描述
- 音频时长，默认 16 秒
- 全景正前方偏移 `yaw_offset_deg`，默认 0°
- 最大区域声源数量，默认 5
- 是否允许人声，默认 false
- 是否允许音乐，默认 false

### 1.2 输出

每个任务输出一个目录：

```text
outputs/<scene_id>/
├── scene_plan.json
├── scene_foa.wav
├── scene_foa.metadata.json
├── panorama_analysis.jpg
├── raw_audio/
│   ├── global_ambience.mp3
│   ├── source_00.mp3
│   └── ...
└── stems/
    ├── global_ambience.wav
    ├── source_00.wav
    └── ...
```

`scene_foa.wav` 必须满足：

```text
通道数：4
采样率：48000 Hz
格式：WAV
默认 subtype：PCM_24
内部处理：float32
Ambisonics convention：AmbiX
通道排列：ACN
归一化：SN3D
通道顺序：[W, Y, Z, X]
```

注意：普通 WAV 头通常不会完整声明 ACN/SN3D 语义，所以必须同步输出 `scene_foa.metadata.json`，明确记录 convention、channel order、normalization、坐标系和生成参数。

### 1.3 交互假设

本 MVP 只生成一个固定参考点上的 Ambisonics 声场：

- 用户转头时，未来播放器可以旋转/解码声场。
- 用户平移时，声音内容、方向、距离和响度不变化。
- 本阶段不实现播放器。
- 本阶段不实现真实 6DoF 音频。

---

## 2. 明确不做的范围

以下内容不得进入本次 MVP：

```text
Marble
HunyuanWorld
3D Gaussian Splatting
深度估计
三维点云
3D 声源坐标
距离衰减
空气吸收
遮挡和绕射
房间反射
实时重编码
SAM3
X-Decoder
SAM2
全景分割
分割投票
HRTF / HRIR 双耳解码
Three.js / WebXR
前端编辑器
移动声源
视频同步
```

原因：当前目标只是生成一条静态四声道 FOA 文件。VLM 给出粗略方向已经足够验证产品闭环。

---

## 3. 与 SonoWorld 上游的关系

截至本文日期，上游仓库已经公开：

- GPT 全景图理解阶段
- MMAudio Text-to-Audio 阶段
- SAM3 全景分割与投票阶段
- Marble 手工接入阶段
- Point / Cluster / Omni 空间配置阶段

但上游 README 仍将完整环境安装、最终 rendering code、interactive viewer 和 evaluation tools 列为待发布内容。因此，本项目不要试图依赖其完整运行环境。

可以参考但不要强耦合的文件：

```text
sonoworld/stages/understanding/gpt.py
sonoworld/stages/audio_generation/mmaudio.py
sonoworld/schemas/understanding.py
sonoworld/utils/audio_source_utils.py
configs/prompts/default_uncond.txt
```

推荐策略：

- 借鉴 `gpt.py` 的“图像 → 声音计划”职责。
- 不沿用旧式自由文本 JSON 解析；改用 Pydantic Structured Outputs。
- 借鉴 `mmaudio.py` 的“每个声音单独生成、转单声道、统一采样率”思想。
- 默认不加载 MMAudio；使用 API provider，减少 CUDA 和模型安装成本。
- 将所有代码放入独立子项目，避免上游未完成依赖影响 MVP。

---

## 4. 最小数据管线

```text
2:1 panorama.jpg
        │
        ▼
输入检查与分析图缩放
        │
        ▼
VLM Scene Planner
        │
        ▼
scene_plan.json
- 1 个 global ambience
- 最多 5 个 regional sources
- 每个区域的 center_u / center_v
- spread / gain / prompt
        │
        ▼
Text-to-Audio Provider
        │
        ▼
独立声音文件
- raw MP3
- mono 48 kHz WAV stem
        │
        ▼
Audio Post-processing
- 解码
- 转 mono
- 重采样到 48 kHz
- 去 DC
- 统一时长
- 峰值基准化
- 应用 gain_db
        │
        ▼
FOA ACN/SN3D Encoder
        │
        ▼
每个声音对应一个 N×4 数组
        │
        ▼
求和、统一缩放、防 clipping
        │
        ▼
scene_foa.wav
```

---

## 5. 项目目录结构

请创建以下结构：

```text
panorama_foa_mvp/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── .env.example
├── prompts/
│   └── panorama_scene_planner.txt
├── src/
│   └── panorama_foa/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── schemas.py
│       ├── pipeline.py
│       ├── image_utils.py
│       ├── coordinates.py
│       ├── planner/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── openai_vlm.py
│       │   └── manual.py
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── provider_base.py
│       │   ├── elevenlabs.py
│       │   ├── mock.py
│       │   ├── decode.py
│       │   └── processing.py
│       └── ambisonics/
│           ├── __init__.py
│           ├── foa.py
│           ├── mixer.py
│           └── exporter.py
├── tests/
│   ├── fixtures/
│   │   ├── panorama_2x1.jpg
│   │   └── manual_plan.json
│   ├── test_coordinates.py
│   ├── test_foa_encoder.py
│   ├── test_audio_processing.py
│   ├── test_exporter.py
│   └── test_pipeline_mock.py
└── examples/
    ├── manual_plan.example.json
    └── README.md
```

---

## 6. Python 与依赖

使用 Python 3.11 或更高版本。

`pyproject.toml` 至少包含：

```text
numpy
scipy
soundfile
pillow
pydantic>=2
openai
httpx
python-dotenv
typer
rich
pytest
```

系统依赖：

```text
ffmpeg
```

原因：ElevenLabs 默认可返回 MP3，需要可靠地解码为 PCM WAV。

启动时检查：

```python
shutil.which("ffmpeg")
```

若缺失，给出明确错误，而不是在音频生成完成后静默失败。

不要引入：

```text
torch
torchaudio
CUDA
SAM3
MMAudio
librosa
```

MMAudio 仅作为未来可选 provider，不是本次交付内容。

---

## 7. 配置和环境变量

`.env.example`：

```dotenv
OPENAI_API_KEY=
OPENAI_VISION_MODEL=
ELEVENLABS_API_KEY=
ELEVENLABS_SOUND_MODEL=eleven_text_to_sound_v2
```

要求：

- 不在日志中打印 API key。
- `OPENAI_VISION_MODEL` 必须可配置，代码中不要依赖永远固定的模型名称。
- 若未提供 `OPENAI_VISION_MODEL`，给出清晰提示，或使用项目 README 中声明的默认值。
- 测试不得依赖真实 API key。

---

## 8. 场景计划 Schema

使用 Pydantic v2 定义严格 Schema。

建议模型：

```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class GlobalAmbience(BaseModel):
    id: str = "global_ambience"
    label: str
    prompt: str
    gain_db: float = Field(ge=-40.0, le=0.0)
    loop: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class RegionalSource(BaseModel):
    id: str
    label: str
    prompt: str
    center_u: float = Field(ge=0.0, le=1.0)
    center_v: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0, le=1.0)
    gain_db: float = Field(ge=-40.0, le=0.0)
    loop: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class ScenePlan(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    scene_description: str
    duration_seconds: float = Field(ge=0.5, le=30.0)
    global_ambience: GlobalAmbience
    regional_sources: list[RegionalSource] = Field(max_length=5)
```

Pipeline 在 VLM 返回后还必须执行二次业务校验：

1. 删除 `confidence < 0.60` 的区域声源。
2. 区域声源最多保留 5 个。
3. 按视觉可信度和预期声学重要性排序。
4. 合并明显重复的标签。
5. 始终只保留一个 global ambience。
6. 若区域声源为空，也允许只输出全局环境声。
7. CLI 的 `duration_seconds` 覆盖 VLM 返回值，避免模型自行改变任务时长。

---

## 9. VLM 规划方案

### 9.1 为什么不用 SAM3

本 MVP 不需要像素级 Mask。VLM 直接返回声源中心的归一化全景坐标：

```text
center_u：从左到右，范围 0～1
center_v：从上到下，范围 0～1
```

再由确定性代码转换为方向。

FOA 的空间分辨率有限，而且第一版用户可通过后续编辑 JSON 修正方向。因此直接使用 VLM 足够完成验证。

### 9.2 图像处理

保留原始全景图；另生成一个仅供 VLM 分析的副本：

```text
最大宽度：4096
保持 2:1
JPEG quality：90
```

输入比例检查：

```text
abs(width / height - 2.0) <= 0.05
```

不符合时默认失败。提供清晰提示：输入必须是完整等距柱状全景图，而不是普通透视图。

### 9.3 OpenAI 调用要求

使用 Responses API 的图像输入能力，并使用 Pydantic Structured Outputs，而不是从 Markdown 代码块中手工提取 JSON。

实现时优先采用当前 OpenAI Python SDK 支持的模式：

```python
response = client.responses.parse(
    model=model_name,
    input=[...],
    text_format=ScenePlan,
)
plan = response.output_parsed
```

将本地图片编码为 Base64 data URL，作为 `input_image` 传入。

若当前 SDK 的图像 + `responses.parse` 组合有接口差异，Codex 应根据已安装 SDK 的官方类型提示调整，但最终必须保留：

- Responses API
- image input
- Pydantic Schema
- 严格验证

### 9.4 VLM Prompt

创建 `prompts/panorama_scene_planner.txt`，内容应表达以下规则：

```text
You are designing a static first-order Ambisonics soundscape from a single
2:1 equirectangular panorama.

The image center represents the visual forward direction. The left and right
edges meet at the back seam. center_u is normalized from image left=0 to
image right=1. center_v is normalized from top=0 to bottom=1.

Identify sounds that are visually present or strongly implied. Prefer
continuous environmental sounds suitable for a static 16-second soundscape.
Do not invent dialogue, music, alarms, animals, vehicles, machines, weather,
or crowds without visual evidence.

Return exactly one subtle global ambience and at most five regional sources.
A regional source is not a 3D point. It is only a broad direction in the
panorama.

For each regional source:
- choose the visual center as center_u and center_v;
- spread=0 means narrowly directional;
- spread=1 means nearly omnidirectional;
- use larger spread for roads, rivers, crowds, forests, surf, and wide areas;
- use smaller spread for compact visible emitters;
- prompts must describe one isolated sound layer;
- prompts should request natural, clean, mono-compatible sound;
- avoid music and intelligible speech unless explicitly allowed;
- make loop=true for continuous ambience;
- gain_db is a relative mix suggestion, normally between -26 and -8 dB.

Merge one physical region crossing the panorama seam into one source. Do not
create both a regional source and a duplicate global source for the same
sound.
```

将用户参数追加到 prompt：

```text
requested duration
optional scene description
allow speech
allow music
maximum regional sources
```

---

## 10. 全景坐标转换

采用固定坐标系：

```text
正前方：azimuth = 0°
左侧：azimuth = +90°
右侧：azimuth = -90°
后方：azimuth = ±180°
水平面：elevation = 0°
头顶：elevation = +90°
脚下：elevation = -90°
```

从归一化像素位置转换：

```python
def normalized_panorama_to_angles(
    center_u: float,
    center_v: float,
    yaw_offset_deg: float = 0.0,
) -> tuple[float, float]:
    azimuth_deg = -360.0 * (center_u - 0.5)
    azimuth_deg += yaw_offset_deg
    azimuth_deg = ((azimuth_deg + 180.0) % 360.0) - 180.0

    elevation_deg = 90.0 - 180.0 * center_v
    elevation_deg = max(-90.0, min(90.0, elevation_deg))
    return azimuth_deg, elevation_deg
```

必须测试：

| center_u | center_v | 预期方向 |
|---:|---:|---|
| 0.50 | 0.50 | 正前方 0° / 0° |
| 0.25 | 0.50 | 左侧 +90° |
| 0.75 | 0.50 | 右侧 -90° |
| 0.00 | 0.50 | 后方 -180° 或 +180° |
| 0.50 | 0.00 | 上方 +90° elevation |
| 0.50 | 1.00 | 下方 -90° elevation |

---

## 11. Text-to-Audio Provider

### 11.1 Provider 接口

```python
from pathlib import Path
from typing import Protocol


class TextToAudioProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        loop: bool,
        output_path: Path,
    ) -> Path:
        ...
```

至少实现：

```text
ElevenLabsSoundEffectsProvider
MockTextToAudioProvider
```

### 11.2 ElevenLabs 实现

调用：

```text
POST https://api.elevenlabs.io/v1/sound-generation
```

Headers：

```text
xi-api-key: <ELEVENLABS_API_KEY>
Content-Type: application/json
```

请求体：

```json
{
  "text": "...",
  "loop": true,
  "duration_seconds": 16,
  "prompt_influence": 0.4,
  "model_id": "eleven_text_to_sound_v2"
}
```

约束：

- `duration_seconds` 限定 0.5～30 秒。
- 默认 `prompt_influence=0.4`，允许配置。
- 默认返回 MP3，保存到 `raw_audio/`。
- 请求超时至少 120 秒。
- 对 429 和 5xx 最多重试 3 次，指数退避。
- 对 4xx 参数错误立即失败并显示响应正文，但不得显示 API key。
- 每个声音单独调用一次 API。
- 最大调用次数：1 个 global + 5 个 regional = 6 次。

推荐声音 prompt 自动追加：

```text
isolated environmental sound layer,
natural clean recording,
mono-compatible,
no added music,
no intelligible speech,
minimal unrelated background sounds
```

但不要重复追加用户已经明确写入的约束。

### 11.3 Mock Provider

测试环境不得调用付费 API。

Mock provider 根据 source index 或 prompt hash 生成确定性的合成信号，例如：

- global ambience：低幅白噪声，经简单低通
- source 0：220 Hz 正弦
- source 1：330 Hz 正弦
- source 2：440 Hz 正弦

要求：

- 输出固定采样率和固定时长。
- 相同 prompt 产生相同结果。
- 峰值不超过 0.5。
- 足以运行端到端测试。

---

## 12. 音频解码与后处理

每个 API 音频依次执行：

```text
MP3/raw bytes
→ ffmpeg 解码为 mono float WAV
→ soundfile 读取 float32
→ 必要时 scipy.signal.resample_poly 到 48 kHz
→ 删除 DC
→ 修正到目标长度
→ 基准峰值归一化
→ 应用 gain_db
→ 保存 stem WAV
```

### 12.1 ffmpeg 命令参考

```bash
ffmpeg -y \
  -i input.mp3 \
  -ac 1 \
  -ar 48000 \
  -c:a pcm_f32le \
  output.wav
```

必须使用 `subprocess.run(..., check=True, capture_output=True)` 并给出可读错误。

### 12.2 单声道

如果输入意外为多声道：

```python
mono = np.mean(audio, axis=1)
```

不要保留模型自带的立体声定位，否则会与 FOA 编码冲突。

### 12.3 去 DC

```python
mono = mono - np.mean(mono)
```

### 12.4 时长处理

目标样本数：

```python
target_samples = round(duration_seconds * 48000)
```

- 过长：直接裁剪。
- 过短且差距小于 250 ms：尾部补零。
- 明显过短：循环复制并使用 100 ms 交叉淡化，直到达到目标长度。

由于 ElevenLabs 可直接请求目标时长，正常情况下不应大量循环。

### 12.5 Stem 基准化

为了让 VLM 的 `gain_db` 具有相对意义：

1. 检查非静音。
2. 将每个 stem 基准峰值缩放到 `-6 dBFS`。
3. 再应用计划中的 `gain_db`。

```python
reference_peak = 10 ** (-6.0 / 20.0)
layer_gain = 10 ** (gain_db / 20.0)
```

若 stem 近似静音，抛出错误并指出具体 source id。

本阶段不要求 LUFS 归一化或复杂 limiter。

---

## 13. FOA Ambisonics 编码

### 13.1 Convention

必须使用：

```text
AmbiX
ACN channel order
SN3D normalization
First Order Ambisonics
[W, Y, Z, X]
```

不要使用：

```text
FuMa WXYZ
N3D
1/sqrt(2) 的 FuMa W 缩放
```

### 13.2 区域声源编码

对单声道信号 `s(t)`：

```text
directionality = 1 - spread

W = s
Y = s × directionality × sin(azimuth) × cos(elevation)
Z = s × directionality × sin(elevation)
X = s × directionality × cos(azimuth) × cos(elevation)
```

其中角度在计算前转换为弧度。

实现：

```python
from __future__ import annotations

import numpy as np


def encode_regional_foa(
    mono: np.ndarray,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    spread: float,
) -> np.ndarray:
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    directionality = 1.0 - float(np.clip(spread, 0.0, 1.0))

    w = signal
    y = signal * directionality * np.sin(azimuth) * np.cos(elevation)
    z = signal * directionality * np.sin(elevation)
    x = signal * directionality * np.cos(azimuth) * np.cos(elevation)

    return np.stack([w, y, z, x], axis=1).astype(np.float32)
```

### 13.3 Global ambience

第一版使用 W-only：

```python
def encode_global_ambience(mono: np.ndarray) -> np.ndarray:
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    output = np.zeros((signal.shape[0], 4), dtype=np.float32)
    output[:, 0] = signal
    return output
```

这表示一个不带明确方向的全向背景层。

### 13.4 Spread 语义

```text
spread = 0.0：方向最明确
spread = 0.5：宽区域方向
spread = 1.0：退化为 W-only 全向层
```

此 spread 是 MVP 的简化参数，不宣称是严格的高阶声源宽度模型。

### 13.5 混合与防 clipping

所有 FOA 数组逐样本相加：

```python
foa_mix = sum(encoded_layers)
```

完成后，计算全部通道的绝对峰值：

```python
peak = np.max(np.abs(foa_mix))
target_peak = 10 ** (-1.0 / 20.0)
if peak > target_peak:
    foa_mix *= target_peak / peak
```

必须对全部通道应用同一个缩放系数，不能分别归一化四个通道，否则会破坏声场比例。

不使用逐样本 hard clip 作为正常增益管理手段。

---

## 14. WAV 导出与 metadata

使用 `soundfile.write`：

```python
sf.write(
    output_path,
    foa_mix,
    samplerate=48000,
    subtype="PCM_24",
    format="WAV",
)
```

`scene_foa.metadata.json` 示例：

```json
{
  "schema_version": "1.0",
  "file": "scene_foa.wav",
  "sample_rate": 48000,
  "channels": 4,
  "ambisonics": {
    "order": 1,
    "convention": "AmbiX",
    "channel_ordering": "ACN",
    "normalization": "SN3D",
    "channel_labels": ["W", "Y", "Z", "X"]
  },
  "coordinates": {
    "azimuth_zero": "front",
    "positive_azimuth": "left",
    "positive_elevation": "up",
    "yaw_offset_deg": 0.0
  },
  "listener_model": {
    "translation_changes_audio": false,
    "rotation_can_change_decode": true
  }
}
```

---

## 15. CLI

使用 Typer 实现。

### 15.1 自动规划并生成

```bash
cd panorama_foa_mvp
python -m panorama_foa.cli generate \
  --panorama ../test-inputs/panorama.jpg \
  --output ../outputs/park_001 \
  --duration 16 \
  --audio-provider elevenlabs \
  --yaw-offset 0
```

### 15.2 Mock 端到端测试

```bash
python -m panorama_foa.cli generate \
  --panorama tests/fixtures/panorama_2x1.jpg \
  --output /tmp/panorama_foa_test \
  --planner manual \
  --plan tests/fixtures/manual_plan.json \
  --audio-provider mock
```

### 15.3 只生成计划

```bash
python -m panorama_foa.cli plan \
  --panorama ../test-inputs/panorama.jpg \
  --output ../outputs/park_001/scene_plan.json
```

### 15.4 使用已有计划重新渲染

```bash
python -m panorama_foa.cli render \
  --plan ../outputs/park_001/scene_plan.json \
  --output ../outputs/park_001 \
  --audio-provider elevenlabs
```

CLI 必须返回非零 exit code 处理以下情况：

- 输入不是有效图像
- 不是约 2:1 全景图
- 缺少 API key
- VLM Schema 验证失败
- Text-to-Audio 请求失败
- ffmpeg 缺失或解码失败
- 音频近似静音
- 输出不是四通道

---

## 16. Pipeline 行为

`pipeline.py` 建议提供：

```python
class PanoramaToFOAPipeline:
    def plan(...): ...
    def generate_stems(...): ...
    def render_foa(...): ...
    def run(...): ...
```

### 16.1 运行顺序

```text
validate panorama
create analysis image
obtain or load ScenePlan
validate/filter ScenePlan
save scene_plan.json
for each audio layer:
    generate raw audio
    decode and post-process
    save mono stem
encode global ambience
encode regional sources
mix all FOA layers
apply one global peak scale
write scene_foa.wav
write metadata
read output back and verify
```

### 16.2 可恢复性

MVP 实现简单缓存：

- 若 raw audio 已存在且 `--force` 未设置，不重新调用 API。
- 若 stem 已存在且 raw audio 未变化，不重新解码。
- 若 scene plan 已存在，`render` 命令可以跳过 VLM。

缓存键建议包含：

```text
provider
model_id
prompt
duration_seconds
loop
```

可以将 hash 写入每个素材旁边的 JSON，但不要为缓存系统增加复杂数据库。

---

## 17. 测试要求

### 17.1 坐标测试

`test_coordinates.py`：

- 图像中心映射到 front。
- 左四分之一映射到 +90°。
- 右四分之一映射到 -90°。
- 上边映射到 +90° elevation。
- yaw offset 正确 wrap 到 `[-180, 180)`。

### 17.2 FOA 方向测试

使用全 1 的 mono 信号，`spread=0`：

#### Front

```text
azimuth=0, elevation=0
W=1, Y=0, Z=0, X=1
```

#### Left

```text
azimuth=+90, elevation=0
W=1, Y=1, Z=0, X≈0
```

#### Right

```text
azimuth=-90, elevation=0
W=1, Y=-1, Z=0, X≈0
```

#### Back

```text
azimuth=180, elevation=0
W=1, Y≈0, Z=0, X=-1
```

#### Up

```text
azimuth 任意, elevation=+90
W=1, Y≈0, Z=1, X≈0
```

#### Full spread

```text
spread=1
W=1, Y=0, Z=0, X=0
```

### 17.3 混音测试

- 多个 layer 输出 shape 为 `(N, 4)`。
- 所有通道使用同一个最终缩放系数。
- 最终峰值不超过 -1 dBFS 的允许误差。
- 输入 stem 不被就地修改。

### 17.4 导出测试

写文件后重新读取并断言：

```text
samplerate == 48000
channels == 4
frames == round(duration * 48000)
subtype == PCM_24，或读取库报告的对应格式
```

### 17.5 端到端测试

使用：

```text
ManualPlanPlanner
MockTextToAudioProvider
```

不联网完成：

```text
panorama → stems → scene_foa.wav → metadata
```

断言所有预期文件存在。

### 17.6 命令

```bash
pytest -q
```

所有测试必须通过。

---

## 18. Manual Plan Fixture

创建 `tests/fixtures/manual_plan.json`：

```json
{
  "schema_version": "1.0",
  "scene_description": "Synthetic directional test panorama",
  "duration_seconds": 2.0,
  "global_ambience": {
    "id": "global_ambience",
    "label": "quiet background",
    "prompt": "quiet diffuse environmental background",
    "gain_db": -24.0,
    "loop": true,
    "confidence": 1.0
  },
  "regional_sources": [
    {
      "id": "front_source",
      "label": "front test source",
      "prompt": "steady isolated front test tone",
      "center_u": 0.5,
      "center_v": 0.5,
      "spread": 0.0,
      "gain_db": -12.0,
      "loop": true,
      "confidence": 1.0
    },
    {
      "id": "left_source",
      "label": "left test source",
      "prompt": "steady isolated left test tone",
      "center_u": 0.25,
      "center_v": 0.5,
      "spread": 0.2,
      "gain_db": -14.0,
      "loop": true,
      "confidence": 1.0
    }
  ]
}
```

---

## 19. README 要求

子项目 README 必须覆盖：

1. 该工具做什么。
2. 它生成的是固定听音中心 FOA，不是真 6DoF。
3. 输入必须是 2:1 equirectangular panorama。
4. 安装 Python 和 ffmpeg。
5. 配置 API keys。
6. 自动生成示例。
7. manual plan + mock 示例。
8. 输出文件解释。
9. ACN/SN3D `[W,Y,Z,X]` 说明。
10. 已知限制。
11. 如何运行测试。
12. 上游 SonoWorld attribution。

不要在 README 中声称：

- 完整复现 SonoWorld
- 物理准确声学
- 真实用户平移音频
- 精确声源距离
- 像素级声源定位

---

## 20. AGENTS.md 要求

在子项目根目录写 `AGENTS.md`，供后续 Codex 继续开发，至少包含：

```text
- Preserve ACN/SN3D channel order [W,Y,Z,X].
- Never normalize FOA channels independently.
- Tests must use mock providers and must not spend API credits.
- Do not add SAM3, Marble, 3DGS, or a frontend without an explicit task.
- Keep API providers behind interfaces.
- Keep input/output schemas backward compatible within schema_version 1.x.
- Every change to coordinate conventions requires updating tests and metadata.
```

---

## 21. 推荐实施顺序

### Commit 1：工程骨架

```text
pyproject
package structure
CLI placeholder
schemas
README skeleton
AGENTS.md
```

### Commit 2：坐标和 FOA 核心

```text
coordinates.py
foa.py
mixer.py
exporter.py
unit tests
```

这一阶段不接任何 API，先确保方向和四声道导出正确。

### Commit 3：Mock 音频和手工计划

```text
manual planner
mock audio provider
audio processing
end-to-end offline test
```

### Commit 4：ElevenLabs provider

```text
HTTP request
retry
raw MP3
ffmpeg decode
cache
```

### Commit 5：OpenAI VLM planner

```text
image validation/downscale
Responses API
Pydantic structured output
prompt
business filtering
```

### Commit 6：文档和最终验证

```text
README examples
.env.example
error messages
full pytest
sample mock output
```

---

## 22. Definition of Done

只有同时满足以下条件才算完成：

- [ ] clone 上游后可以在 `panorama_foa_mvp/` 独立安装。
- [ ] 不需要安装 SonoWorld 全部模型或 CUDA 依赖。
- [ ] 输入 2:1 全景图可以生成 ScenePlan。
- [ ] 支持从已有 JSON plan 跳过 VLM。
- [ ] ElevenLabs provider 可以生成每层独立素材。
- [ ] Mock provider 可以完全离线运行。
- [ ] 所有 stem 被转换为 mono、48 kHz、相同长度。
- [ ] 输出严格为四通道 WAV。
- [ ] 通道顺序严格是 `[W,Y,Z,X]`。
- [ ] metadata 明确写明 ACN/SN3D/AmbiX。
- [ ] front/left/right/back/up 单元测试通过。
- [ ] 最终四通道使用统一增益缩放，不独立归一化。
- [ ] `pytest -q` 全部通过。
- [ ] README 含完整安装和运行示例。
- [ ] 不包含 Marble、SAM3、分割投票和 3D 依赖。

---

## 23. Codex 最终汇报格式

完成后请输出：

```text
1. 新增和修改的文件列表
2. 架构摘要
3. 关键实现决策
4. 实际执行过的命令
5. 测试结果
6. 一个 mock 端到端命令
7. 一个真实 API 端到端命令
8. 仍存在的限制
```

若真实 API 因缺少 key 无法执行，应明确说明：

```text
API provider 已实现并通过 mock/HTTP mock 测试，但未进行付费线上调用。
```

不要伪造真实 API 成功结果。

---

## 24. 后续版本预留，但本次不实现

接口设计应允许未来加入：

```text
SAM3SegmentationPlanner
MMAudioProvider
用户编辑 scene_plan.json
双耳 preview renderer
Web 播放器
全景图声源可视化
多听音中心
3D 定位和真实 6DoF
```

但这些内容不得阻塞本次 MVP。

---

## 25. 已验证的外部接口与约定

### SonoWorld

```text
https://github.com/HuMathe/sonoworld
```

当前仓库默认包含 GPT understanding、MMAudio generation、SAM3 segmentation 和 spatial config；README 明确表示完整 setup、rendering code 和 viewer 尚未全部发布。

### OpenAI 图像理解与 Structured Outputs

```text
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/structured-outputs
```

使用 Responses API、`input_image` 和 Pydantic Structured Outputs。

### ElevenLabs Sound Effects

```text
https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert
```

当前接口：

```text
POST /v1/sound-generation
model_id=eleven_text_to_sound_v2
duration_seconds: 0.5～30
loop: boolean
prompt_influence: 0～1
```

### AmbiX convention

```text
https://plugins.iem.at/docs/compatibility/
```

使用：

```text
ACN channel ordering
SN3D normalization
FOA order: W, Y, Z, X
```

### 音频文件和重采样

```text
https://python-soundfile.readthedocs.io/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
```

---

## 26. 最终产品定义

本项目不是 3D 声学模拟器。它是：

> 一个把 2:1 全景图转换为语义匹配、方向大致合理的四声道 FOA Ambisonics 环境声文件的离线生成工具。

最终最小闭环：

```text
panorama.jpg
→ VLM scene plan
→ isolated mono sound layers
→ ACN/SN3D FOA encoding
→ scene_foa.wav
```

这已经足以验证：

- 全景视觉内容是否能自动映射到声音清单；
- Text-to-Audio 生成素材是否适合空间化；
- 四声道 FOA 是否能被标准 Ambisonics 工具正确解码；
- 后续是否值得增加 SAM3 和 3D 定位。
