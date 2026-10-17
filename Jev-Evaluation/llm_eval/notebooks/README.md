# JevBench 入门 Notebook

`jevbench_intro.ipynb` 是一个自包含教程，覆盖：

- JevBench **是什么 / 有什么 / 做什么 / 怎么做**
- `llm_eval` 框架的代码结构
- **怎么跑 mock 验证整套流程**（零成本）
- **怎么接真实模型**（自动检测 env var、并行跑所有可用 provider）
- **怎么读懂结果**（summary.json 字段白话解释 + matplotlib 对比柱状图）
- **怎么对比多个模型**（multi-runner + 跨模型对比表 + 持久化图表）

## 怎么打开

| 平台 | 步骤 |
|---|---|
| **本地 Jupyter** | `jupyter lab` 后在文件树里打开 `jevbench_intro.ipynb` |
| **Jupyter Notebook** | `jupyter notebook` |
| **VS Code** | 安装 "Jupyter" 扩展，直接打开 `.ipynb` |
| **Google Colab** | 上传到 Drive / GitHub 后打开 |
| **魔搭 ModelScope** | `Notebook` tab → 上传 `.ipynb` |
| **阿里云 PAI / DSW** | `python3 -m ipykernel install --user` 后开 lab |
| **百度 BML / 腾讯 TI-ONE** | 同样走 IPython magic |

Cell 0 自动检测当前平台并选用合适的 `pip install` 入口。

## 怎么用 API key

真实 API 调用只在 **cell 12** 触发（之前所有 cell 都用 `mock` adapter，零成本）。

两种方式提供 key：

### A. 直接粘贴到 cell 11（**学习用，最方便**）

打开 cell 11，把你的 key 填进 `direct_keys` 字典。Notebook 配置了 6 个 provider：Jev（native）+ 5 个 OpenAI 模型（verbalized，共用一个 key）：

```python
direct_keys = {
    "TYPESAFE_API_KEY": "",   # Jev (TypeSafe System One, native)
    "OPENAI_API_KEY":   "",   # OpenAI — gpt-6.1-sol / gpt-6-luna / gpt-5.6-terra / gpt-5.6-luna / gpt-4.1-mini
}
```

⚠️ **粘贴的 key 会保存到 `.ipynb` 文件里**。请勿：
- 把含 key 的 notebook commit 到 git
- 把含 key 的 notebook 截图 / 拷贝给他人
- 把 notebook 上传到任何公开 / 第三方平台

学习完成后请把 key 删干净并清空 outputs：
- Jupyter: 菜单 → Cell → All Output → Clear
- VS Code: 菜单 → Edit → Clear All Outputs

### B. 环境变量（**生产 / 共享场景**）

```bash
export TYPESAFE_API_KEY="tsk_..."    # Jev
export OPENAI_API_KEY="sk-..."       # 全部 OpenAI 模型
```

Cell 11 / 12 自动检测两路来源，**A 优先于 B**。
Cell 11 的"source"列会显示每个 key 来自 `pasted` 还是 `env`。

### 改 base_url（Azure OpenAI / 代理 / 兼容网关）

每个 provider 默认走官方端点。如需改走 Azure OpenAI、代理或 OpenAI 兼容网关：

**Cell 11 内置粘贴区**（已预填官方默认值）：
```python
base_url_overrides = {
    "TYPESAFE_API_KEY":  "https://api.typesafe.ai",     # Jev (native)
    "OPENAI_API_KEY":    "https://api.openai.com/v1",   # OpenAI
}
```

**或用 env var**（优先级低于 base_url_overrides）：

```bash
export OPENAI_API_KEY_BASE_URL="https://your-gateway.example.com/v1"
```

Cell 11 的检测表会显示**当前生效的 base_url**（如果是覆盖值，会显示覆盖后的，不是默认）。

### 当前 6 个 provider 的价格

价格来自 [OpenAI 官方价目](https://developers.openai.com/api/docs/pricing)（standard tier，短上下文，2026-10-17 核对），
定义在 `llm_eval/adapters/__init__.py` 的 `OPENAI_MODELS`，notebook 与 runner 记账同源。

| provider | input $/M | output $/M | 备注 |
|---|---|---|---|
| jev | 0.042 | 0 | native System One，output 免费 |
| gpt-6.1-sol | 2.00 | 10.00 | 推理模型 |
| gpt-6-luna | 0.10 | 0.50 | 推理模型，`openai` preset 默认 |
| gpt-5.6-terra | 2.00 | 12.00 | 推理模型 |
| gpt-5.6-luna | 0.20 | 1.20 | 推理模型 |
| gpt-4.1-mini | 0.40 | 1.60 | 非推理模型（temperature=0 基线） |

推理模型（gpt-5* / gpt-6* / o 系列）由 adapter 自动改用 `max_completion_tokens`、去掉 `temperature`、
设 `reasoning_effort="low"`，并把输出上限提到 2048（预算 ledger 按同一上限预留）。
如果实际价格变化，编辑 `OPENAI_MODELS` 即可。

## 调参

| Env var | 默认 | 含义 |
|---|---|---|
| `LBEVAL_MAX_WORKERS` | `5` | 并发上限；超过时 cell 12 自动分批提示 |
| `LBEVAL_CAP_USD` | `0.30` | 每 provider 的预算上限 |

## 跑哪一档

Cell 12 默认跑 `tasks/public/easy.jsonl`（48 题）。
改 cell 12 里的 `tasks_path`：

- `tasks/smoke.jsonl` — 6 题（秒级）
- `tasks/public/easy.jsonl` — 48 题（默认）
- `tasks/public_all.jsonl` — 231 题（≈$0.04–$0.05/provider）

## 产物

跑完后所有产物在 `runs/notebook-demo/`：

```
runs/notebook-demo/
├── smoke/                        # cell 8 产物
├── real/<provider>/              # cell 12 产物（每个 provider 一个目录）
│   ├── results.jsonl
│   ├── summary.json
│   ├── ledger.jsonl
│   └── raw/<task_id>.json
├── multi/                        # cell 13-14 产物
│   ├── compare.json
│   ├── compare.md
│   └── charts/
│       ├── 13_mock_compare.png   # cell 13 mock 对比图
│       └── 14_real_compare.png   # cell 14 真实/兜底对比图
└── _tmp/                         # 中间产物（多 runner 的临时编排目录）
```

## 怎么改 notebook

`scripts/build_notebook.py` 是 source-of-truth，里面定义了 17 个 cell 的源码。
改完后：

```sh
python3 scripts/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/jevbench_intro.ipynb \
  --output notebooks/jevbench_intro_executed.ipynb --ExecutePreprocessor.timeout=180
```

## CI 集成（可选）

如果想保证 notebook 不退化，可以加：

```yaml
# .github/workflows/notebook.yml
- run: pip install nbformat jupyter matplotlib
- run: python scripts/build_notebook.py
- run: jupyter nbconvert --to notebook --execute notebooks/jevbench_intro.ipynb --output /tmp/out.ipynb
```

`nbconvert --execute` 失败时 exit code 非零，能直接 fail PR。
但执行 notebook 走 mocked cells（cell 8/9/13）就够——cell 12 在 CI 里**不要**给真实 key。