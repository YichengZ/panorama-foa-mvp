# Codex 启动提示词：多 Agent 开发 Panorama → FOA MVP

请使用当前 Codex 的原生 subagents、项目级 custom agents 和 Git worktrees，完成 `CODEX_HANDOFF_PANORAMA_TO_FOA.md` 定义的项目。不要把本任务当作普通单 agent 编码任务。

## 最高目标

Clone SonoWorld 仓库后，新增一个独立、可安装、可测试的 `panorama_foa_mvp/` 子项目，实现：

```text
2:1 equirectangular panorama
→ VLM static sound plan
→ isolated mono text-to-audio stems
→ deterministic FOA AmbiX encoding
→ 4-channel 48 kHz WAV in ACN/SN3D order [W,Y,Z,X]
```

用户平移不改变声音。本阶段只输出 Ambisonics 文件，不实现 Marble、SAM3、深度、3DGS、点云、HRTF、播放器、前端或真 6DoF。

## 必须先做

1. 完整阅读 `CODEX_HANDOFF_PANORAMA_TO_FOA.md`。
2. Clone 仓库并创建 integration branch。
3. 创建文档要求的 `.codex/config.toml`、`.codex/agents/*.toml`、release-gate skill 和 `panorama_foa_mvp/AGENTS.md`。
4. 创建并提交：
   - `docs/IMMUTABLE_SCOPE.md`
   - `docs/ACCEPTANCE_MATRIX.md`
   - `docs/DECISIONS.md`
5. 在任何功能代码之前，显式 spawn 并等待以下只读 subagents：
   - `repo_explorer`
   - `test_designer`
   - `goal_guardian`
6. 综合三者结果形成 `IMPLEMENTATION_PLAN.md`，再让 `goal_guardian` 独立审核。只有它返回 `GO` 才开始实现。

## 主 agent 角色

你是 conductor 和 integrator，不是唯一开发者。你负责：

- 保持不可变目标；
- 拆分任务并定义不重叠文件所有权；
- 创建或指挥 worktrees；
- 等待和汇总 subagent 结果；
- 按顺序 cherry-pick；
- 运行集成测试；
- 执行发布门禁；
- 最终报告证据。

不要把仓库扫描、长测试日志和逐文件审查堆进主线程。要求 subagent 返回 handoff packet：

```text
STATUS: PASS | FAIL | BLOCKED
SCOPE:
SUMMARY:
EVIDENCE:
FILES_CHANGED:
COMMANDS_RUN:
TESTS:
RISKS:
RECOMMENDED_NEXT_ACTION:
```

## 实现方式

### 只读工作

并行执行探索、测试设计、路线审查和最终 review。只读 agents 不得修改代码。

### 写工作

优先为以下任务建立四个隔离 worktree/background thread：

1. `agent/test-contract` → `test_worker`，先提交独立验收测试，不得修改生产代码
2. `agent/foa-core` → `foa_worker`
3. `agent/offline-pipeline` → `pipeline_worker`
4. `agent/api-providers` → `api_worker`

严格遵守 handoff 文档中的文件所有权。写 agents 不得编辑同一个文件，也不得修改独立 acceptance tests。

若当前 Codex 环境不能让这些 agent 使用隔离 worktree，不要让它们在同一 checkout 并行写。改为按顺序执行：

```text
test_worker → foa_worker → pipeline_worker → api_worker
```

只读 agents 仍可并行。

## 测试合同

在实现代码前，显式启动 `test_worker`，根据 `test_designer` 输出在独立 worktree 中提交 acceptance tests。允许初始失败，但实现 agents 不得：

- 删除测试；
- 放宽容差以掩盖错误；
- 改变坐标系期望；
- 允许逐通道归一化；
- 让测试访问真实付费 API；
- 用 mock 覆盖本应测试的数学逻辑。

任何验收合同变更必须先经 `goal_guardian` 明确批准并记录到 `DECISIONS.md`。

## 集成顺序

```text
测试合同
→ FOA core
→ offline pipeline
→ API providers
→ 文档
```

每次 cherry-pick 后运行相应测试。集成完成后运行完整 `pytest -q` 和 mock CLI 端到端任务，并独立回读生成的 WAV。

## 强制发布门禁

集成后显式并行 spawn：

- `goal_guardian`
- `acceptance_tester`
- `independent_reviewer`

等待三者全部完成。不要让一个 reviewer 的结论提前影响其他 reviewer。之后再运行 Codex `/review` 对集成分支 diff 做额外独立检查。

只有同时满足以下条件才能宣布完成：

- goal guardian 最终 PASS；
- acceptance tester 的 mandatory rows 全部 PASS；
- 没有未解决的 P0/P1；
- `pytest -q` 通过；
- mock 端到端通过；
- 输出 WAV 是 4 通道、48 kHz、精确帧数；
- metadata 声明 AmbiX、ACN、SN3D、`[W,Y,Z,X]`；
- front/left/right/back/up 与 spread 测试通过；
- 四通道只使用统一最终缩放；
- 测试中没有真实付费 API 调用；
- 没有引入明确排除的 3D、分割、播放或前端范围。

如果门禁失败，把 finding 拆成最小修复任务交给合适的 worker。Reviewer agents 不得自己修复。修复后完整重跑发布门禁，不接受只重测失败项。

## 不得做

- 不要完整复现 SonoWorld。
- 不要引入 Agents SDK、Symphony 或额外 orchestration service；当前项目使用 Codex 原生能力已经足够。
- 不要引入 SAM3、Marble、HunyuanWorld、3DGS、深度、点云、HRTF 或 Web UI。
- 不要让 Text-to-Audio 直接生成四声道；必须先生成单声道 stems，再由确定性代码编码和混合。
- 不要伪造真实 API 测试结果。
- 不要因为“以后可能需要”扩展当前范围。

## 最终报告

最终只在所有门禁完成后报告：

1. 架构和实现摘要；
2. 新增/修改文件；
3. 每个 agent 的最终状态；
4. 所有 worktree branch 和 commit SHA；
5. 实际运行的命令；
6. pytest 结果；
7. mock 端到端结果；
8. WAV 检查结果；
9. 真实 API 是否实际执行；
10. goal guardian、acceptance tester、independent reviewer 和 `/review` 的结论；
11. acceptance matrix 的逐项证据；
12. 已知限制。

现在开始执行。不要等待额外澄清；遇到非关键歧义时使用 handoff 文档中的默认值，并记录到 `DECISIONS.md`。
