```python
# Cell 0 — Platform detection & dependency install
import importlib
import os
import subprocess
import sys
from pathlib import Path

def detect_platform() -> str:
    if 'google.colab' in sys.modules:
        return 'colab'
    if 'PAI' in os.environ or 'aliyun' in os.environ.get('HOSTNAME', ''):
        return 'aliyun_pai'
    img = os.environ.get('JUPYTER_IMAGE', '').lower()
    if 'modelscope' in img:
        return 'modelscope'
    if 'bml' in os.environ.get('HOSTNAME', '').lower():
        return 'baidu_bml'
    if 'tione' in os.environ.get('HOSTNAME', '').lower():
        return 'tencent_tione'
    return 'jupyter_local'

PLATFORM = detect_platform()
print(f"platform: {PLATFORM}")
print(f"python:   {sys.version.split()[0]}")

# Run from the project root (the dir holding the llm_eval package, tasks/ and runs/);
# the notebook itself lives under notebooks/.
NOTEBOOK_DIR = Path.cwd().resolve()
for ancestor in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]:
    if (ancestor / "llm_eval" / "__init__.py").exists():
        if ancestor != NOTEBOOK_DIR:
            os.chdir(ancestor)
            print(f"cd -> {ancestor}")
        break

def install(pkg: str, quiet: bool = True) -> None:
    flag = '-q' if quiet else ''
    if PLATFORM == 'colab':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', flag, pkg])
    else:
        from IPython import get_ipython
        get_ipython().run_line_magic('pip', f'install {flag} {pkg}')

# Install llm_eval from this checkout, and matplotlib for charts.
for pkg, name in [('-e .', 'llm_eval'), ('matplotlib', 'matplotlib')]:
    try:
        m = importlib.import_module(name)
        print(f"  ok: {name} already installed")
    except ImportError:
        print(f"  installing {name}...")
        install(pkg, quiet=True)
        importlib.import_module(name)
        print(f"  ok: {name} installed")

# Matplotlib backend-safe: avoid font warnings on minimal containers.
# Imported here, not at the top: the loop above may have just installed it.
import matplotlib  # noqa: E402
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
print("matplotlib backend:", matplotlib.get_backend())
```

# Cell 1 — JevBench 入门

## JevBench 入门：结构化决策模型评测

> 25 分钟从"听过 JevBench"走到"我的 API 已经跑出第一份结果并读懂"。

**JevBench** 是一个专门评测**结构化决策模型**的基准测试，对应 *"typed decision model"* 这种用法：
给定一段事实 (state) 和受限于固定选项集 (rubric + label set)，模型必须返回对**精确选项集**的概率分布——
不写自由文本，不做文字解释。

### 这个 notebook 给谁看

| 你想做的事 | 你应该看 |
|---|---|
| 理解 JevBench 是什么 / 不是什么 | Cell 1–3, 5, 7 |
| 不消耗 API 跑通流程 | Cell 0–9, 13 |
| 用自己的 API 实测 | Cell 10–12 |
| 同时测多个模型对比 | Cell 12 + 14 |
| 接入 Colab / 魔搭 / 阿里云 PAI | Cell 0 (auto-detect) |

### 默认 zero-cost

- **所有示例用 `mock` adapter**，不需要任何 API key。
- **Cell 12 才是真实 API 调用**——它会自动检测你 `export` 了哪些 key，没 key 时优雅跳过。
- **不会写你的 key 进任何文件**，也不会 echo 到日志。

按 Shift+Enter 一路往下跑就行。

# Cell 2 — 什么是 JevBench

## 什么是 JevBench

JevBench 评测的不是"模型有多能说"，而是"模型能不能稳定地做**对固定选项集的判定**"。
它围绕五条互相约束的轴：

| 轴 | 关心什么 | 典型问题 |
|---|---|---|
| **smart** | 答得对不对 | accuracy、Brier、ECE、calibration |
| **cheap** | 跑一次判定多少钱 | $/1k decisions (按 provider 自报 tariff × 实际 token) |
| **fast** | 多快 | p50 / p95 端到端延迟（含网络） |
| **reliable** | 输出能不能信任 | schema_validity、paraphrase consistency、置信度可不可信 |
| **open** | 权重 / 代码能不能用 | 三项独立事实：代码 license、权重是否可下、权重 license |

合成方式（v1.3）：**JevBench Score = 几何平均(Intelligence 30%, Calibration 25%, Speed 25%, Cost 25%)**，Intelligence < 50 时按 `(I/50)²` 折扣。

### 六族题

| 族 | 任务 |
|---|---|
| `routing` | 客服请求分发 |
| `adequacy` | 答案合格度判定 |
| `policy` | yes/no 合规检查 |
| `intent` | 意图分类 |
| `ordinal` | 严重度评分 |
| `extraction` | 字段抽取 |

### 它跟 MMLU / LMSYS / HELM 的区别

- **MMLU**: 多选题，正确率单维度。
- **LMSYS Arena**: 聊天偏好，人类投票。
- **HELM**: 多维度横扫，但偏学术 benchmark。
- **JevBench**: 唯一**专为"输出必须是结构化概率分布"的模型**设计，并把**延迟和成本**作为一等公民。

当你的业务需要的是"判定 + 概率"而不是"对话 + 推理"时，JevBench 是更直接的对照表。

# Cell 3 — 为什么需要专门的决策模型评测

## 为什么需要专门的决策模型评测

很多生产场景要的**不是聊天，而是判定**：

- 客服请求路由到哪个部门
- 答案合格度评分（生成式模型的输出质量门控）
- 内容是否合规的 yes/no 判定
- 用户的真实意图分类
- 工单严重度 0–3 打分
- 字段抽取（订单号 / 地址 / 金额）

把这些场景硬塞给通用 LLM 的"自由文本回答"，下游就得做正则解析、做边界 case、做错误恢复——**而且得不到 calibration**。
`{A: 0.7, B: 0.2, C: 0.1}` 这种概率分布天然适合：
- 阈值化决策（argmax）
- 拒绝 / 转人工（低置信度）
- 集成 / 串行 / 多模型仲裁

但**"让通用 LLM 写出一个合法的概率分布"和"模型原生支持这种 API"是两件事**。
JevBench 把这件事显式分开：
- `native` 分布：模型 API 本身就给概率分布（TypeSafe 的 Jev、djev、kev 等）
- `verbalized` 分布：让通用 LLM 用 JSON schema 写出来（精度和稳定性都吃亏）

跑同一份题，两类模型得分能直接对比。


```python
# Cell 4 — 环境自检
import importlib
import sys

print("Python:", sys.version.split()[0])

required = [
    ("llm_eval", "the benchmark harness we built"),
    ("matplotlib", "for charts"),
    ("json", "stdlib, but just confirming"),
]
missing = []
for name, desc in required:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "stdlib")
        print(f"  ✓ {name:12s} {v}  ({desc})")
    except ImportError:
        missing.append(name)
        print(f"  ✗ {name:12s} MISSING  ({desc})")

if missing:
    print("\nFix: re-run cell 0 (it installs llm_eval and matplotlib from this checkout).")
else:
    print("\nAll required packages present.")
```

# Cell 5 — 本框架的结构

## 本框架 (`llm_eval`) 的结构

```
llm_eval/
├── task.py          题目 dataclass + 校验 + 哈希
├── scorer.py        分布校验 + argmax + Brier + MAE（fail-closed）
├── metrics.py       aggregate：accuracy / brier / ece / 延迟 / cost / paraphrase
├── ledger.py        fcntl 文件锁 + append-only 预算 ledger
├── runner.py        串行驱动 adapter；401/429/连续错停跑
├── summarize.py     public_export() 重算 + 白名单
├── multi_config.py  JSON 多 runner 配置加载
├── multi_runner.py  多 runner 编排
├── compare.py       跨模型汇总表
├── cli.py           run / summarize / multi-run / compare 4 个子命令
└── adapters/        各家 API 的薄壳
        base.py            AdapterError + HTTP helper
        openai_compat.py   OpenAI Chat Completions 实现（含推理模型参数适配）
        typesafe.py        Jev / TypeSafe native System One
        mock.py            不消耗 API 的伪响应（always right / uniform）
tasks/
├── smoke.jsonl                      6 题（1 paraphrase 对）
└── public/{easy,original,hard}.jsonl   JevBench v1.2 MIT 公开题 231 道
multi_run.example.json               多 runner 配置模板
tests/                              63 个单元测试
```

设计纪律（也写在 `IMPLEMENTATION.md` 里）：
- **不重试、不重映射、不修补**坏的响应
- **reserve-before / settle-after** 真实计费
- **公共导出** = 白名单 + 重算，永不 echo raw request / response
- **adapter 内部串行**——并发只在 notebook 这层做（cell 12）


```python
# Cell 6 — 浏览题库
from pathlib import Path
from collections import Counter
import json

from llm_eval.task import load_tasks

repo = Path.cwd()
print(f"repo root: {repo}\n")

print("=== task files ===")
for p in sorted((repo / "tasks").rglob("*.jsonl")):
    rel = p.relative_to(repo)
    n = sum(1 for _ in p.open())
    print(f"  {rel}  ({n} tasks)")

print("\n=== load public_all.jsonl (231 tasks) ===")
tasks = load_tasks(repo / "tasks" / "public_all.jsonl")
print(f"loaded: {len(tasks)}")

by_family = Counter(t.family for t in tasks)
by_topic = Counter(t.topic for t in tasks)
by_qtype = Counter(t.question_type for t in tasks)

print("\nby family:")
for k, v in sorted(by_family.items(), key=lambda kv: -kv[1]):
    print(f"  {k:14s} {v:3d}")

print("\nby topic:")
for k, v in sorted(by_topic.items(), key=lambda kv: -kv[1]):
    print(f"  {k:18s} {v:3d}")

print("\nby question_type:")
for k, v in sorted(by_qtype.items(), key=lambda kv: -kv[1]):
    print(f"  {k:8s} {v:3d}")

# Show 2 full records: one easy + one hard (per PRD §C)
print("\n=== sample 1 (easy: intent classification) ===")
easy = next(t for t in tasks if t.family == "intent" and t.question_type == "choice")
print(json.dumps(easy.to_dict(), indent=2, ensure_ascii=False))

print("\n=== sample 2 (hard: long policy reasoning) ===")
hard = next(t for t in tasks if "long_policy" in t.id)
sample = hard.to_dict()
sample["state"] = {**sample["state"], "text": sample["state"].get("text", "")[:200] + "...[truncated for display]"}
print(json.dumps(sample, indent=2, ensure_ascii=False))
```

# Cell 7 — 9 个指标

## JevBench 评测什么 — 9 个指标

| 指标 | 范围 | 一句话解释 |
|---|---|---|
| **accuracy** | [0, 1] | 模型 argmax 答对的比例。对照 `majority_class_accuracy` 才有意义（不然 60% 看起来不错，但永远选 `yes` 就是 82%）。 |
| **`majority_class_accuracy`** | [0, 1] | 永远选最高频 label 能拿多少分。每个 family 都有偏斜（adequacy 是 82%），accuracy 必须对照这个下限读。 |
| **`schema_validity`** | [0, 1] | 响应能被解析成合法概率分布的比例。**headline 容忍度 2%**（允许 `0.999` 这种舍入误差）。 |
| **`schema_validity_strict`** | [0, 1] | 冻结的 1e-3 容忍度。两个都报：headline 容忍度 vs 严格数字。 |
| **Brier** | [0, 2] | 多类 `Σ (p_k − y_k)²`；二元题用 2-class 约定让数字可比。**越低越好**。 |
| **ECE** | [0, 1] | 期望校准误差。10 个等宽 bin 按 top-label 置信度分；空 bin 缺席而非 0。**越低越好**。 |
| **`ordinal_mae`** | [0, N−1] | 序数题上概率加权期望值 vs 参考 level 的 MAE。和 argmax accuracy 是**两个不同预测**，同时报。 |
| **`paraphrase_consistency`** | [0, 1] | 同一题的两种问法：(a) 答得一样、(b) 都答对。**两个都报**——只报"答得一样"会让"两个都错"的高分。 |
| **latency p50 / p95** | seconds | 端到端延迟（含网络）。首个请求单独报，scale-to-zero 端点把冷启动算在第一个头上。 |
| **cost** | USD / 1k decisions | `derived_usage_times_tariff`（实测 token × provider 自报 tariff）。无计费账户写 `null` + 显式 basis 标签（如 `no_billable_account_public_endpoint`）。**单位是"每千次判定"，不是每千 token**。 |

每个指标的代码实现在：
- `llm_eval/scorer.py` — 单题打分（含分布校验、argmax、Brier、MAE）
- `llm_eval/metrics.py` — 聚合（ECE 10-bin、paraphrase consistency、latency p50/p95、cost）
- `llm_eval/summarize.py` — 公共导出白名单 + 重算


```python
# Cell 8 — 第一次跑：mock + smoke
import json
import os
from pathlib import Path

from llm_eval.adapters import get_adapter
from llm_eval.runner import run
from llm_eval.summarize import public_export
from llm_eval.task import load_tasks

# 输出统一落在 runs/notebook-demo/
OUT_ROOT = Path("runs/notebook-demo/smoke")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

tasks = load_tasks("tasks/smoke.jsonl")
adapter = get_adapter(name="mock")  # default: 永远答对，confident

results = run(
    tasks=tasks,
    adapter=adapter,
    ledger_path=str(OUT_ROOT / "ledger.jsonl"),
    cap_usd=1.0,
    out_path=str(OUT_ROOT / "results.jsonl"),
    raw_dir=str(OUT_ROOT / "raw"),
    price_in_per_m=0.0,    # mock 免费
    price_out_per_m=0.0,
    delay_s=0.0,
    max_input_tokens=2000,
    max_output_tokens=200,
)

summary = public_export(results, tasks, run_meta={"cell": 8, "adapter": "mock"})
with open(OUT_ROOT / "summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"completed: {len(results)} tasks")
print(f"output:    {OUT_ROOT}")
print(f"  - results.jsonl   ({len(results)} records)")
print("  - summary.json    (public-safe aggregate)")
print("  - ledger.jsonl    (cost events)")
print(f"  - raw/*.json      ({len(list((OUT_ROOT/'raw').iterdir()))} raw req/resp bodies)")
```


```python
# Cell 9 — 读懂 summary.json
import json

summary = json.load(open("runs/notebook-demo/smoke/summary.json"))

# 白话注释
field_notes = {
    "n_tasks": "题目总数",
    "n_scorable": "有 expected label 可打分的题数（= n_tasks 当全部题都有 expected）",
    "accuracy": "模型答对的比例",
    "majority_class_accuracy": "永远选最高频 label 的下限（baseline）",
    "schema_validity": "headline 容忍度（2%）下合法的比例",
    "schema_validity_strict": "严格容忍度（1e-3）下合法的比例",
    "brier": "校准损失，越低越好",
    "ece": "期望校准误差，越低越好",
    "ordinal_mae": "序数题的概率加权 MAE",
    "p50_s": "中位数延迟",
    "p95_s": "P95 延迟",
    "cost_usd_total": "总成本（USD）",
    "paraphrase_consistency": "paraphrase 对的同答 / 都对率",
    "per_family": "按 family 拆分的 accuracy",
    "per_topic": "按 topic 拆分的 accuracy",
    "run_meta": "本轮运行的元数据",
}

print(f"{'key':30s} {'value':20s}  note")
print("-" * 90)
for k in ["n_tasks", "n_scorable", "accuracy", "majority_class_accuracy",
         "schema_validity", "schema_validity_strict", "brier", "ece",
         "ordinal_mae", "p50_s", "p95_s", "cost_usd_total"]:
    v = summary.get(k)
    if isinstance(v, float):
        vstr = f"{v:.4f}"
    else:
        vstr = str(v)
    note = field_notes.get(k, "")
    print(f"{k:30s} {vstr:20s}  {note}")

print()
print("per_family:")
for fam, v in sorted(summary.get("per_family", {}).items()):
    print(f"  {fam:14s} n={v['n']:3d}  acc={v['accuracy']:.4f}")

print()
print("per_topic:")
for top, v in sorted(summary.get("per_topic", {}).items()):
    print(f"  {top:18s} n={v['n']:3d}  acc={v['accuracy']:.4f}")

print()
print("paraphrase_consistency:", summary.get("paraphrase_consistency"))
```

# Cell 10 — 接真实模型：环境准备

## 接真实模型：环境准备

### 两种方式提供 key（任选其一）

#### A. 直接在 cell 11 粘贴（**仅学习用，最方便**）

打开 cell 11，把你的 key 填进 `direct_keys` 字典。**注意**：填了的 key 会保存到 `.ipynb` 文件里，请勿把含 key 的 notebook 提交到 git 或分享给他人。

#### B. 用环境变量（**生产 / 共享场景**）

```bash
export TYPESAFE_API_KEY="tsk_..."        # Jev (TypeSafe)
export OPENAI_API_KEY="sk-..."           # OpenAI — 一个 key 跑全部 5 个 OpenAI 模型
```

Cell 11 自动检测两路来源，**A 优先于 B**。

### 安全纪律

- **Cell 11 顶部有红色警告**——粘贴 key 前请先读。
- **Cell 11 / 12 不 echo key**——检测结果只显示"set / not set"。
- **cap_usd 限制**：`LBEVAL_CAP_USD`（默认 0.30/provider）防止意外爆额度。
- **并发上限**：`LBEVAL_MAX_WORKERS`（默认 5）；超过 5 个 key 时 cell 12 会分批提示。
- **如果用了 A 方式**，运行前 `git diff notebooks/` 应该**没有**新增内容（`notebooks/*_executed.ipynb` 已在 .gitignore 里，但 canonical `.ipynb` 没有）。

### 跑哪一档

cell 12 默认跑 `tasks/public/easy.jsonl`（48 题），成本约 $0.001–$0.05/provider。
如果你想跑全部 231 题，改 cell 12 里的 `tasks_path` 到 `tasks/public_all.jsonl`。



```python
# Cell 11 — 检查 & 配置 API key
import os

from llm_eval.adapters import OPENAI_MODELS

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ ⚠️  DANGER ZONE — 明文 key 粘贴区（仅学习用！）                          │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 在这里直接填 key 会保存到 .ipynb 文件里。请勿：                          │
# │   ✗ 把含 key 的 notebook commit 到 git                                  │
# │   ✗ 把含 key 的 notebook 截图 / 拷贝给他人                              │
# │   ✗ 把 notebook 上传到任何公开 / 第三方平台                              │
# │                                                                          │
# │ 学习完成后请把这里填的 key 删干净，并清空 outputs：                       │
# │   菜单 → Cell → All Output → Clear  (Jupyter)                          │
# │   菜单 → Edit → Clear All Outputs  (VS Code)                           │
# └─────────────────────────────────────────────────────────────────────────┘

# 2 个 key — Jev (native System One) + OpenAI（一个 key 跑全部 OpenAI 模型，verbalized）
direct_keys = {
    "TYPESAFE_API_KEY": "",   # Jev (TypeSafe System One, native)
    "OPENAI_API_KEY": "",     # OpenAI — gpt-6.1-sol / gpt-6-luna / gpt-5.6-terra / gpt-5.6-luna / gpt-4.1-mini
}

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ ↑ 替换上面的空字符串为你的 key，例如：                                    │
# │    "TYPESAFE_API_KEY":  "tsk_...",                                       │
# │    "OPENAI_API_KEY":    "sk-...",                                        │
# └──────────────────────────────────────────────────────────────────────────┘

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ base_url 覆盖区 ───                                                      │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ 留空 = 用 provider 官方默认地址                                          │
# │ Azure OpenAI / 代理 / 兼容网关用户：在 OPENAI_API_KEY 的位置填 base_url  │
# └──────────────────────────────────────────────────────────────────────────┘

base_url_overrides = {
    # 官方直连 URL（默认值，可不填）：
    "TYPESAFE_API_KEY":  "https://api.typesafe.ai",          # Jev (native)
    "OPENAI_API_KEY":    "https://api.openai.com/v1",        # OpenAI
}

# 合并 key：direct_keys > env var
effective = {}
for k, v in direct_keys.items():
    effective[k] = v if v else os.environ.get(k, "")

# 合并 base_url：base_url_overrides > env var _BASE_URL > 默认
def resolve_base_url(key_env: str, default_url: str) -> str:
    override = base_url_overrides.get(key_env, "")
    if override:
        return override
    env_url = os.environ.get(key_env + "_BASE_URL", "")
    if env_url:
        return env_url
    return default_url

# 当前 6 个 provider 配置 — Jev (native System One) + 5 个 OpenAI 模型 (verbalized)
PROVIDER_TABLE = [
    # Jev — TypeSafe System One (native)。output tokens free，output 价格 = 0。
    {"name": "jev",                 "adapter": "jev",      "model": "jev-latest",
     "key_env": "TYPESAFE_API_KEY",
     "default_base_url": "https://api.typesafe.ai",
     "price_in_per_m": 0.042, "price_out_per_m": 0.0,
     "mode": "native",
     "notes": "Jev (TypeSafe native System One)"},
] + [
    # OpenAI verbalized 模型 — 通用 LLM 用 JSON schema 写出概率分布。
    # 价格来自 llm_eval.adapters.OPENAI_MODELS（与 runner 记账同源）。
    {"name": model,                 "adapter": "openai",   "model": model,
     "key_env": "OPENAI_API_KEY",
     "default_base_url": "https://api.openai.com/v1",
     "price_in_per_m": price["price_in_per_m"], "price_out_per_m": price["price_out_per_m"],
     "mode": "verbalized",
     "notes": f"OpenAI {model}"}
    for model, price in OPENAI_MODELS.items()
]

# 计算 effective base_url
for p in PROVIDER_TABLE:
    p["base_url"] = resolve_base_url(p["key_env"], p["default_base_url"])

print(f"{'provider':22s} {'key':6s} {'source':8s} {'base_url':50s}")
print('-' * 95)
for p in PROVIDER_TABLE:
    val = effective.get(p["key_env"], "")
    has = bool(val)
    mark = "✓" if has else "✗"
    src = "pasted" if direct_keys.get(p["key_env"]) else ("env" if os.environ.get(p["key_env"]) else "-")
    url_short = p["base_url"]
    if len(url_short) > 48:
        url_short = url_short[:45] + "..."
    note = p.get("notes", "")
    print(f"  {mark}  {p['name']:20s} {('set' if has else '-'):6s} {src:8s} {url_short:50s} {note}")

MAX = int(os.environ.get("LBEVAL_MAX_WORKERS", "5"))
CAP = os.environ.get("LBEVAL_CAP_USD", "1.50")
print(f"\nLBEVAL_MAX_WORKERS = {MAX} (cell 12 will batch if more keys are detected)")
print(f"LBEVAL_CAP_USD      = {CAP} (per-provider budget cap; default 1.50 for full benchmark)")

# 安全检查：如果用了 pasted 方式，提醒一次
any_pasted = any(direct_keys.values())
if any_pasted:
    print("\n⚠️  You have pasted plaintext keys above. Remember:")
    print("    - Do NOT commit this notebook if it has keys filled in")
    print("    - Do NOT share or upload this notebook")
    print("    - Clear outputs and remove keys before saving")

```


```python
# Cell 12 — 跑真模型：并行所有可用 provider（≤5 一批，跑全量 benchmark）
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request

from IPython.display import display, Markdown

from llm_eval.adapters import get_adapter
from llm_eval.multi_config import MultiRunConfig, RunnerConfig
from llm_eval.multi_runner import run_multi
from llm_eval.task import load_tasks

# PROVIDER_TABLE 复用 cell 11 的（含 base_url / price / request_options）。
# 如果你直接执行本 cell 而跳过了 cell 11，下面有一份简化副本（不会自动用粘贴 key）。
if "PROVIDER_TABLE" not in globals():
    from llm_eval.adapters import OPENAI_MODELS
    PROVIDER_TABLE = [
        # Jev — native System One（TypeSafe），output free
        {"name": "jev",              "adapter": "jev",      "model": "jev-latest",
         "key_env": "TYPESAFE_API_KEY",
         "default_base_url": "https://api.typesafe.ai",
         "price_in_per_m": 0.042, "price_out_per_m": 0.0,
         "mode": "native"},
    ] + [
        # OpenAI — verbalized（让通用 LLM 用 JSON schema 写出概率分布）
        {"name": model,              "adapter": "openai",   "model": model,
         "key_env": "OPENAI_API_KEY",
         "default_base_url": "https://api.openai.com/v1",
         "price_in_per_m": price["price_in_per_m"], "price_out_per_m": price["price_out_per_m"],
         "mode": "verbalized"}
        for model, price in OPENAI_MODELS.items()
    ]
    # fallback base_url & default request_options
    for p in PROVIDER_TABLE:
        p.setdefault("base_url", p["default_base_url"])
        p.setdefault("request_options", None)
        p.setdefault("skip_reason", "")

MAX_PARALLEL = min(int(os.environ.get("LBEVAL_MAX_WORKERS", "5")), 5)
CAP_USD = float(os.environ.get("LBEVAL_CAP_USD", "1.50"))
TASKS_PATH = os.environ.get("LBEVAL_TASKS_PATH", "tasks/public_all.jsonl")  # 231 题全量

OUT_ROOT = Path("runs/notebook-demo/real")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 加载 tasks 用于预估成本 + 真实 token 校验
all_tasks = load_tasks(TASKS_PATH)
n_tasks = len(all_tasks)
print(f"benchmark: {TASKS_PATH}  ({n_tasks} tasks)")
print("  estimated per-provider cost (input tokens × price):")
for p in PROVIDER_TABLE:
    if p.get("skip_reason"):
        continue
    if p["price_in_per_m"] is not None:
        est_in = n_tasks * 1500 * p["price_in_per_m"] / 1e6
        est_out = n_tasks * 50 * (p["price_out_per_m"] or 0) / 1e6
        print(f"    {p['name']:18s}  ~${est_in + est_out:.4f}  (in {est_in:.4f} + out {est_out:.4f})")

# Key 来源合并：cell 11 的 `direct_keys` 优先，回退到 env var
_effective_keys = {}
_base_url_overrides: dict[str, str] = {}
if "direct_keys" in globals():
    for k, v in direct_keys.items():
        _effective_keys[k] = v if v else os.environ.get(k, "")
    _base_url_overrides = dict(globals().get("base_url_overrides", {}))
else:
    for p in PROVIDER_TABLE:
        _effective_keys[p["key_env"]] = os.environ.get(p["key_env"], "")


def _resolve_base_url(p: dict) -> str:
    """优先级：cell 11 base_url_overrides > env <KEY>_BASE_URL > default"""
    if _base_url_overrides.get(p["key_env"]):
        return _base_url_overrides[p["key_env"]]
    env_url = os.environ.get(p["key_env"] + "_BASE_URL", "")
    if env_url:
        return env_url
    return p.get("default_base_url", p.get("base_url", ""))


def _chat_probe(key: str, base_url: str, model: str, adapter_name: str = "") -> int:
    """1-shot probe 提前检测 401/402/404。返回 HTTP status。

    - 默认（OpenAI 兼容）：POST /chat/completions with {model, messages, max_tokens:1}
    - Jev (native System One)：GET /v1/models（只要 key 正确就 200；systemone 需要合法 state）
    """
    if adapter_name in ("jev", "typesafe", "openai"):
        # Jev has no /chat/completions — auth-check via /v1/models.
        # OpenAI: GET /models/{model} is free and returns 404 for an unknown
        # model; a 1-token chat probe would be rejected by reasoning models.
        if adapter_name == "openai":
            url = base_url.rstrip("/") + "/models/" + model
        else:
            url = base_url.rstrip("/") + "/v1/models"
        try:
            req = urllib.request.Request(url, method="GET",
                headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 0
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }).encode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


# Step 1 — 扫可用 provider；probe 每家区分 401/402/404/200
available = []
skipped = []
for p in PROVIDER_TABLE:
    if p.get("skip_reason"):
        skipped.append((p, f"config: {p['skip_reason']}"))
        continue
    key = _effective_keys.get(p["key_env"], "").strip()
    if not key:
        skipped.append((p, f"no {p['key_env']}"))
        continue
    p["base_url"] = _resolve_base_url(p)

    code = _chat_probe(key, p["base_url"], p["model"], adapter_name=p["adapter"])
    if code == 200:
        available.append(p)
        print(f"  ✓  {p['name']:18s}  base_url={p['base_url']}  model={p['model']}")
    elif code in (401, 403):
        skipped.append((p, f"http {code}: auth rejected (check key validity / plan)"))
    elif code == 402:
        skipped.append((p, f"http {code}: plan 额度用尽（去 {p['key_env'].replace('_API_KEY','').lower()} 续费）"))
    elif code == 404:
        skipped.append((p, f"http {code}: model '{p['model']}' 此 endpoint 不可用"))
    elif code == 0:
        skipped.append((p, "chat endpoint unreachable（DNS / TLS / 代理）"))
    else:
        skipped.append((p, f"http {code}: 未知状态"))

if skipped:
    print("\n" + f"skipped {len(skipped)} provider(s):")
    for p, reason in skipped:
        print(f"  ✗  {p['name']:18s}  {reason}")

print(f"\nrunning {len(available)} provider(s) × {n_tasks} tasks, "
      f"max parallel={MAX_PARALLEL}, cap=${CAP_USD} per provider")

if not available:
    display(Markdown(
        "⚠️ **No API keys available** (or all probes failed).\n\n"
        "Two ways to provide keys:\n\n"
        "**A. Paste directly in cell 11** (`direct_keys` dict) — convenient for learning, "
        "but the key is saved into this notebook file.\n\n"
        "**B. `export OPENAI_API_KEY=...` in your shell** before launching Jupyter — "
        "safer for production.\n\n"
        "Without working keys, cell 14 has no real data to plot. Re-run this cell after fixing the listed skips."
    ))
else:
    # Step 2 — 分批
    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    batches = list(chunked(available, MAX_PARALLEL))

    # 所有 OpenAI 模型共用一个 key：在开线程前一次性写入 env，线程里不再增删，
    # 避免一个线程跑完把 key 删掉、另一个线程读不到。
    for p in available:
        os.environ[p["key_env"]] = _effective_keys[p["key_env"]]

    all_runs = []
    for batch_idx, batch in enumerate(batches):
        print("\n=== batch " + f"{batch_idx+1}/{len(batches)}: {[p['name'] for p in batch]}" + " ===")

        def run_one(spec):
            try:
                cfg = MultiRunConfig(
                    run_id=f"cell12-{spec['name']}",
                    tasks_path=TASKS_PATH,
                    runners=[RunnerConfig(
                        name=spec["name"], adapter=spec["adapter"], model=spec["model"],
                        endpoint=spec["base_url"],
                        key_env=spec["key_env"],   # ← 关键修复：之前是 ""，runner 取不到 key
                        price_in_per_m=spec["price_in_per_m"],
                        price_out_per_m=spec["price_out_per_m"],
                        skip_if_done=True,
                    )],
                    cap_usd=CAP_USD,
                    delay_s=0.3,
                    out_root=str(OUT_ROOT / "_tmp"),
                    request_options=spec.get("request_options"),
                )
                manifest = run_multi(cfg)
                return (spec["name"], "ok", manifest)
            except Exception as e:
                return (spec["name"], "error", repr(e))

        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = [ex.submit(run_one, p) for p in batch]
            for fut in futures:
                name, status, payload = fut.result()
                if status == "ok":
                    # multi_runner 会把 name slug 化成下划线（dot→underscore），
                    # 用 manifest dir 取最稳，不要硬编码 name。
                    runner_dir = Path(payload["runners"][0]["dir"])
                    summary_path = runner_dir / "summary.json"
                    if summary_path.exists():
                        import shutil
                        target = OUT_ROOT / name
                        target.mkdir(parents=True, exist_ok=True)
                        for fname in ("results.jsonl", "summary.json", "ledger.jsonl"):
                            src = runner_dir / fname
                            if src.exists():
                                shutil.copy(src, target / fname)
                        src_raw = runner_dir / "raw"
                        if src_raw.is_dir():
                            shutil.copytree(src_raw, target / "raw", dirs_exist_ok=True)
                        print(f"  [done] {name}: copied to {target}")
                    all_runs.append(name)
                else:
                    print(f"  [error] {name}: {str(payload)[:200]}")

        if batch_idx + 1 < len(batches):
            remaining = [p["name"] for p in available[batch_idx+1:]]
            display(Markdown(
                f"⚠️ **还有 {len(remaining)} 个 provider 没跑**：{remaining}。\n\n"
                f"再次执行本 cell 跑下一批（当前上限 {MAX_PARALLEL}）。"
                f"`LBEVAL_MAX_WORKERS` 可调。"
            ))

    print(f"\nbatch summary: {len(all_runs)} provider(s) ran successfully")
    print(f"output root: {OUT_ROOT}")

# === Troubleshooting 子节 ===
# 常见情况：
# 1. summary.schema_validity 偏低（< 0.7）：
#    - 模型返回了概率但某些题没覆盖所有 label → 长 label 集上 schema 漏报
#    - 模型有时返回空字符串 → rate-limit，或推理模型把输出预算全花在推理上
# 2. summary.accuracy 高但 brier/ece 也高：
#    - 答对了但置信度没校准——自信错的也自信对。
# 3. 大量 error 记录 → adapter 没适配该 provider 的 response_format；改 llm_eval/adapters/openai_compat.py
# 4. cost_usd_total = None → 该 provider 没设 price，框架拒绝"编"价格
# 5. 直填 key 后报错"key empty"：cell 11 里 `"OPENAI_API_KEY": ""` 是空字符串就视作未填，
#    填一个真 key 即可（任意非空字符串都能识别为 "set"）。
# 6. base_url 覆盖不生效：检查 cell 11 的 base_url_overrides 字典 key 是否跟 key_env 一致。
# 7. OpenAI 模型报 404：该 key 所在的 project 没开通这个模型；在 platform.openai.com 的 project 设置里启用。
# 8. 推理模型（gpt-5* / gpt-6* / o 系列）报 `unsupported parameter`：adapter 已自动改用
#    max_completion_tokens、去掉 temperature，并设 reasoning_effort="low"；
#    若某模型不支持该值，在 cell 11 的 PROVIDER_TABLE 里给它加 "request_options": {"reasoning_effort": None}。
# 9. OpenAI 报 429：rate limit / 额度不足；调小 LBEVAL_MAX_WORKERS 或检查 billing。


```


```python
# Cell 13 — Mock 锚点对照（cell 14 会消费这里产出的 compare.json）
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Re-imported so this cell also runs on its own (e.g. after a kernel restart).
from llm_eval.multi_config import MultiRunConfig, RunnerConfig  # noqa: F811
from llm_eval.multi_runner import run_multi  # noqa: F811
from llm_eval.compare import compare_run, write_markdown_table

# 两个 mock 锚点：mock-perfect（理论满分） + mock-uniform（随机分布）
# 都是 cell 14 横评里的上下界参照系；真实 provider 由 cell 12 注入。
TASKS = "tasks/smoke.jsonl"  # cell 13 用 smoke（秒级）；cell 14 真实 provider 跑的是 public_all（231 题）

cfg = MultiRunConfig(
    run_id="cell13-multi-mock",
    tasks_path=TASKS,
    runners=[
        RunnerConfig(name="mock-perfect", adapter="mock", model="perfect",   key_env=""),
        RunnerConfig(name="mock-uniform", adapter="mock", model="uniform-1", key_env=""),
    ],
    cap_usd=1.0, delay_s=0.0,
    out_root="runs/notebook-demo/multi/_tmp",
)
manifest = run_multi(cfg)
print(f"runners: {len(manifest['runners'])}")
for r in manifest["runners"]:
    print(f"  {r['name']:20s} status={r['status']:8s} n_ok={r.get('n_ok','-')}")

# Compare
cmp = compare_run("runs/notebook-demo/multi/_tmp/cell13-multi-mock")
charts_dir = Path("runs/notebook-demo/multi/charts")
charts_dir.mkdir(parents=True, exist_ok=True)
write_markdown_table(cmp, str(charts_dir.parent / "compare.md"))
(cmp_path := charts_dir.parent / "compare.json").write_text(
    json.dumps(cmp, ensure_ascii=False, indent=2)
)

# === 内联展示全指标对比表（不只"已导出"） ===
rows = cmp["rows"]
cols = [
    ("adapter/model", lambda r: f"{r.get('adapter','')}/{r.get('model','')}"),
    ("accuracy",     lambda r: r.get("accuracy")),
    ("majority_base",lambda r: r.get("majority_class_accuracy")),
    ("schema_valid", lambda r: r.get("schema_validity")),
    ("schema_strict",lambda r: r.get("schema_validity_strict")),
    ("brier",        lambda r: r.get("brier")),
    ("ece",          lambda r: r.get("ece")),
    ("ordinal_mae",  lambda r: r.get("ordinal_mae")),
    ("p50_s",        lambda r: r.get("p50_s")),
    ("p95_s",        lambda r: r.get("p95_s")),
    ("cost_usd",     lambda r: r.get("cost_usd_total")),
]
def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"
    return str(v)

header = "| runner | " + " | ".join(c[0] for c in cols) + " |"
sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
lines = [header, sep]
for name, row in rows.items():
    cells = [name] + [fmt(c[1](row)) for c in cols]
    lines.append("| " + " | ".join(cells) + " |")
display(Markdown("### Mock multi-model comparison (cell 13)\n\n" + "\n".join(lines)))

# === 3x3 图：全 9 指标对比 ===
metrics_9 = [
    ("accuracy",          "Accuracy (↑)",          "linear", 1.05),
    ("schema_validity",   "Schema valid (↑)",      "linear", 1.05),
    ("schema_validity_strict","Schema strict (↑)", "linear", 1.05),
    ("brier",             "Brier (↓)",             "log",    None),
    ("ece",               "ECE (↓)",               "log",    None),
    ("ordinal_mae",       "Ordinal MAE (↓)",       "linear",  None),
    ("p50_s",             "p50 latency (↓)",       "log",    None),
    ("p95_s",             "p95 latency (↓)",       "log",    None),
    ("cost_usd_total",    "Cost / 1k decisions (↓)","log",   None),
]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
names = list(rows.keys())
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
for idx, (key, title, scale, ymax) in enumerate(metrics_9):
    r, c = divmod(idx, 3)
    ax = axes[r, c]
    vals = [(rows[n].get(key) or 0) for n in names]
    ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))])
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=20)
    if scale == "log":
        positive = [v for v in vals if v > 0]
        if len(positive) >= 1:
            ax.set_yscale("log")
    if ymax is not None:
        ax.set_ylim(0, ymax)
fig.suptitle("JevBench full-metric comparison (cell 13 mock)", fontsize=14)
fig.tight_layout()
chart_path = charts_dir / "13_full_metrics.png"
fig.savefig(chart_path, dpi=120, bbox_inches='tight')
display(Markdown(f"**chart saved**: `{chart_path}`"))
plt.close(fig)

# 也保存老 4 联图
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
metrics_4 = [
    ("accuracy",      [rows[n].get("accuracy", 0) for n in names],      "Accuracy (↑)"),
    ("brier",         [rows[n].get("brier", 0) or 0 for n in names],     "Brier (↓)"),
    ("p50_s",         [rows[n].get("p50_s", 0) or 0 for n in names],     "p50 (s)"),
    ("cost_usd_total",[rows[n].get("cost_usd_total") or 0 for n in names], "Cost USD (↓)"),
]
for ax, (k, v, t) in zip(axes, metrics_4):
    ax.bar(names, v, color=[colors[i % len(colors)] for i in range(len(names))])
    ax.set_title(t)
    ax.tick_params(axis='x', rotation=15)
    if k in ("brier", "p50_s", "cost_usd_total") and any(x > 0 for x in v):
        ax.set_yscale("log")
fig.suptitle("JevBench 4-key comparison (cell 13 mock)", fontsize=13)
fig.tight_layout()
fig.savefig(charts_dir / "13_mock_compare.png", dpi=120, bbox_inches='tight')
plt.close(fig)

display(Markdown(
    "\n所有产物：\n"
    "- `runs/notebook-demo/multi/compare.json`\n"
    "- `runs/notebook-demo/multi/compare.md`\n"
    "- `runs/notebook-demo/multi/charts/13_full_metrics.png` (3×3 全指标)\n"
    "- `runs/notebook-demo/multi/charts/13_mock_compare.png` (4 联简版)"
))
```


```python
# Cell 14 — Main score + 4 维横评（token / time / cost / accuracy）
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from llm_eval.task import load_tasks

# 严格只显示真实数据（cell 12 runner 写到磁盘的 summary.json / results.jsonl + cell 13 mock 锚点）。
# 缺失数据 → 显示 '—'，绝不编造任何数字。

def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) >= 100 else f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)

def _fmt_usd(v):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"${v:.4f}"
    return str(v)

def _load_real_tokens(results_path: Path):
    """Sum prompt_tokens / completion_tokens from results.jsonl."""
    if not results_path.exists():
        return None, None
    in_t = out_t = 0
    any_usage_seen = False
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            usage = rec.get("usage") or {}
            it = usage.get("prompt_tokens", 0) or 0
            ot = usage.get("completion_tokens", 0) or 0
            if it or ot:
                any_usage_seen = True
            in_t += it
            out_t += ot
    if not any_usage_seen:
        return None, None
    return in_t, out_t

# === Step 1: 扫描真实 runs (cell 12 写到磁盘的) ===
real_dir = Path("runs/notebook-demo/real")
real_rows = {}
if real_dir.is_dir():
    for d in sorted(real_dir.iterdir()):
        if not d.is_dir() or d.name == "_tmp":
            continue
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text())
        except Exception:
            continue
        in_t, out_t = _load_real_tokens(d / "results.jsonl")
        real_rows[d.name] = {
            "data_source": "real",
            "accuracy": s.get("accuracy"),
            "p50_s": s.get("p50_s"),
            "p95_s": s.get("p95_s"),
            "token_in": in_t,
            "token_out": out_t,
            "cost_usd_total": s.get("cost_usd_total"),
            "schema_validity": s.get("schema_validity"),
            "brier": s.get("brier"),
            "ece": s.get("ece"),
            "ordinal_mae": s.get("ordinal_mae"),
            "jevbench_score": s.get("jevbench_score"),
        }

n_real = len(real_rows)
src_label = f"{n_real} real runner(s)"

display(Markdown(f"## 4-dim cross-model comparison — {src_label}"))
if n_real == 0:
    display(Markdown(
        "🔴 **No real provider data available.** "
        "Set at least one `*_API_KEY` env var (or paste into cell 11) and re-run cell 12. "
        "The charts / tables below need real API responses to plot — no synthesis, no estimation."
    ))

# === Step 2: 4 维表格 (accuracy / time / token / price) ===
table_cols = [
    ("data_src",   lambda r: r.get("data_source", "—")),
    ("accuracy",   lambda r: _fmt(r.get("accuracy"))),
    ("p50_s (time)", lambda r: _fmt(r.get("p50_s"))),
    ("token_in",   lambda r: _fmt(r.get("token_in"))),
    ("token_out",  lambda r: _fmt(r.get("token_out"))),
    ("cost_usd (price)", lambda r: _fmt_usd(r.get("cost_usd_total"))),
]
header = "| runner | " + " | ".join(c[0] for c in table_cols) + " |"
sep = "|" + "|".join(["---"] * (len(table_cols) + 1)) + "|"
table_lines = [header, sep]
for name, row in sorted(real_rows.items()):
    cells = [name] + [c[1](row) for c in table_cols]
    table_lines.append("| " + " | ".join(cells) + " |")
display(Markdown("\n".join(table_lines)))

# === Step 3: 4 维横评图 (2x2) — accuracy / time / token / price ===
charts_dir = Path("runs/notebook-demo/multi/charts")
charts_dir.mkdir(parents=True, exist_ok=True)

def _bar(ax, names, vals, title, ymax=None):
    real_pos = [v for v in vals if isinstance(v, (int, float)) and v > 0]
    if not real_pos:
        ax.text(0.5, 0.5, "no real data\n(set API key + re-run cell 12)",
                ha="center", va="center", transform=ax.transAxes,
                color="#aa0000", fontsize=11)
        ax.set_title(title, fontsize=11, color="#aa0000")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    disp = [0.0 if v is None else float(v) for v in vals]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    bar_colors = [colors[i % len(colors)] if (isinstance(vals[i], (int, float)) and vals[i] > 0)
                  else "lightgray" for i in range(len(names))]
    bars = ax.bar(names, disp, color=bar_colors)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            bar.set_hatch("//")
            bar.set_edgecolor("gray")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    if title.endswith("(↓)") and len(real_pos) >= 2:
        ax.set_yscale("log")
    if ymax is not None and real_pos:
        ax.set_ylim(0, max(ymax, max(real_pos) * 1.1))

names = sorted(real_rows.keys())
PANELS_4DIM = [
    ("accuracy",        "Accuracy ↑",            1.05),
    ("p50_s",           "p50 latency (time) ↓",  None),
    ("token_in",        "Token in ↓",            None),
    ("cost_usd_total",  "Cost USD (price) ↓",    None),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for idx, (key, title, ymax) in enumerate(PANELS_4DIM):
    r, c = divmod(idx, 2)
    ax = axes[r, c]
    vals = [real_rows[n].get(key) for n in names]
    _bar(ax, names, vals, title, ymax=ymax)
fig.suptitle(f"JevBench 4-dim comparison (real-only data) — {src_label}", fontsize=13)
fig.tight_layout()
chart_path = charts_dir / "14_4dim_compare.png"
fig.savefig(chart_path, dpi=120, bbox_inches='tight')
plt.close(fig)
display(Markdown(f"**4-dim chart saved**: `{chart_path}`\n\n"
                 f"_灰色斜纹 = 该行**无真实数据**（没 key / API 失败 / API 没返回 usage）；实色 = 真实数据_"))

# === Step 4: JevBench composite score ranking (1 张总排名图) ===
# JevBench Score = 几何平均 (Intelligence, Calibration, Speed, Cost) in 0-100,
# with low-Intelligence penalty (I<50 → score * (I/50)²). 见 llm_eval.metrics.jevbench_score.
score_rows = []
for name, row in real_rows.items():
    js = row.get("jevbench_score") or {}
    composite = js.get("score")
    axes_dict = js.get("axes") or {}
    score_rows.append({
        "runner": name,
        "composite": composite,
        "intelligence": axes_dict.get("intelligence"),
        "calibration": axes_dict.get("calibration"),
        "speed": axes_dict.get("speed"),
        "cost": axes_dict.get("cost"),
        "n_scorable": js.get("n_scorable"),
        "penalty": js.get("intelligence_penalty_applied", False),
    })

# Sort by composite descending; None last
score_rows_sorted = sorted(
    score_rows,
    key=lambda r: (r["composite"] is None, -(r["composite"] or 0))
)

display(Markdown("# JevBench Score ranking (composite + 4 axes)"))
if not score_rows_sorted:
    display(Markdown("_No jevbench_score yet — re-run cell 12 after setting keys._"))
else:
    rank_lines = ["| rank | runner | composite | intelligence | calibration | speed | cost | notes |",
                   "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(score_rows_sorted):
        composite = r["composite"]
        composite_str = f"{composite:.1f}" if composite is not None else "—"
        note = "⚠️ I<50, penalty applied" if r["penalty"] else ""
        rank_lines.append(
            f"| {i+1} | {r['runner']} | {composite_str} | "
            f"{_fmt(r['intelligence'])} | {_fmt(r['calibration'])} | "
            f"{_fmt(r['speed'])} | {_fmt(r['cost'])} | {note} |"
        )
    display(Markdown("\n".join(rank_lines)))

# Bar chart: composite score (sorted)
fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(score_rows_sorted) + 1)))
rnames = [r["runner"] for r in score_rows_sorted]
comps = [r["composite"] for r in score_rows_sorted]
real_pos = [v for v in comps if v is not None]
if real_pos:
    colors = ["#2a9d8f" if not r["penalty"] else "#e76f51" for r in score_rows_sorted]
    bars = ax.barh(rnames, [v if v is not None else 0 for v in comps], color=colors)
    for i, (bar, v) in enumerate(zip(bars, comps)):
        width = v if v is not None else 0
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}" if v is not None else "—",
                va='center', fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("JevBench composite score (0-100)")
    ax.set_title(f"JevBench score ranking (real-only) — {src_label}", fontsize=13)
    ax.invert_yaxis()
    fig.tight_layout()
else:
    ax.text(0.5, 0.5, "no real jevbench_score data\n(set API key + re-run cell 12)",
            ha="center", va="center", transform=ax.transAxes, color="#aa0000", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
rank_chart = charts_dir / "14_main_score.png"
fig.savefig(rank_chart, dpi=120, bbox_inches='tight')
plt.close(fig)
display(Markdown(f"**Main-score chart saved**: `{rank_chart}`\n\n"
                 f"_绿色 = 4-axis 满分；红色 = Intelligence<50 触发了 (I/50)² 折扣_"))

# === Step 5: 持久化合并结果 ===
merged_path = Path("runs/notebook-demo/multi/compare_merged.json")
merged_payload = {
    "rows": real_rows,
    "n_real": n_real,
    "label": src_label,
    "ranking": score_rows_sorted,
}
merged_path.write_text(
    json.dumps(merged_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
display(Markdown(
    f"\n所有产物：\n"
    f"- `{merged_path}`（合并视图，仅真实数据，{n_real} 行）\n"
    f"- `runs/notebook-demo/multi/charts/14_4dim_compare.png`（4 维横评：accuracy / p50 latency / token_in / cost）\n"
    f"- `runs/notebook-demo/multi/charts/14_main_score.png`（JevBench 总分排名图）"
))

# === Step 6: Task coverage — 列出每个 provider 实际跑了哪些题 / 没跑哪些 ===
# 加载真实 task 集（默认是 tasks/public_all.jsonl = 231 题）
TASKS_PATH = globals().get("TASKS_PATH", "tasks/public_all.jsonl")  # cell 12 未在本 session 运行时的回退
_coverage_tasks = load_tasks(TASKS_PATH)
_all_task_ids = [t.id for t in _coverage_tasks]
display(Markdown(f"## Task coverage — 跑了哪些 / 没跑哪些\n\n"
                 f"全集: **{len(_all_task_ids)} tasks** (`{TASKS_PATH}`)"))

if real_rows:
    coverage_lines = ["| runner | total | ok ✓ | invalid ✗ | error ✗ | unattempted | covered task_ids |",
                      "|---|---|---|---|---|---|---|"]
    for name in sorted(real_rows.keys()):
        # Read this runner's results.jsonl
        rs_path = Path("runs/notebook-demo/real") / name / "results.jsonl"
        # if symlink, follow
        statuses = {"ok": [], "invalid": [], "error": [], "unattempted": []}
        if rs_path.exists():
            with rs_path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    s = r.get("status", "?")
                    if s in statuses:
                        statuses[s].append(r.get("task_id", "?"))
        covered = sorted(set(sum(statuses.values(), [])))
        uncovered = sorted(set(_all_task_ids) - set(covered))
        covered_str = ", ".join(covered[:8])
        if len(covered) > 8:
            covered_str += f" … (+{len(covered)-8} more)"
        coverage_lines.append(
            f"| {name} | {sum(len(v) for v in statuses.values())} | "
            f"{len(statuses['ok'])} | {len(statuses['invalid'])} | "
            f"{len(statuses['error'])} | {len(statuses['unattempted'])} | "
            f"{covered_str or '—'} |"
        )
    display(Markdown("\n".join(coverage_lines)))

    # 对每个 runner 单独显示 ok / invalid / error / unattempted 的 task_id
    display(Markdown("### 哪些跑了 / 没跑 (per runner)"))
    for name in sorted(real_rows.keys()):
        rs_path = Path("runs/notebook-demo/real") / name / "results.jsonl"
        statuses = {"ok": [], "invalid": [], "error": [], "unattempted": []}
        if rs_path.exists():
            with rs_path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    s = r.get("status", "?")
                    if s in statuses:
                        statuses[s].append(r.get("task_id", "?"))
        covered = sorted(set(sum(statuses.values(), [])))
        uncovered = sorted(set(_all_task_ids) - set(covered))
        ok_n = len(statuses['ok'])
        err_n = len(statuses['error'])
        unat_n = len(statuses['unattempted'])
        inv_n = len(statuses['invalid'])

        md_lines = [f"#### `{name}` — covered **{len(covered)}/{len(_all_task_ids)}**"]
        md_lines.append(f"- ✅ ok ({ok_n}): {', '.join(statuses['ok'][:5])}{' …' if ok_n > 5 else ''}")
        if inv_n:
            md_lines.append(f"- ⚠️ invalid ({inv_n}): {', '.join(statuses['invalid'][:5])}{' …' if inv_n > 5 else ''}")
        if err_n:
            md_lines.append(f"- ❌ error ({err_n}): {', '.join(statuses['error'][:5])}{' …' if err_n > 5 else ''}")
        if unat_n:
            md_lines.append(f"- ⏸️ unattempted ({unat_n}): {', '.join(statuses['unattempted'][:5])}{' …' if unat_n > 5 else ''}")
        if uncovered:
            uncovered_str = ', '.join(uncovered[:5])
            if len(uncovered) > 5:
                uncovered_str += f' … (+{len(uncovered)-5} more)'
            md_lines.append(f"- 🕳️ **没跑** ({len(uncovered)}): {uncovered_str}")
        if not uncovered:
            md_lines.append("- ✅ 全部题都跑了")
        display(Markdown("\n".join(md_lines)))
else:
    display(Markdown("_No real runner data — skip coverage report._"))
```

# Cell 15 — 故障模式与设计纪律

## 故障模式与设计纪律

`llm_eval` runner 的不变量：

| 规则 | 做什么 | 不做什么 |
|---|---|---|
| **不重试** | 401/403/429 立即停跑 | 不 sleep 重试、不换 endpoint 重试 |
| **不重映射** | 模型答非所问时直接当 invalid | 不"补"成合法分布、不替模型猜 |
| **不修补** | malformed 分布记 invalid 并计 0 分 | 不重 parse、不删多余字段 |
| **reserve-before / settle-after** | 真扣费先 reserve，settle 用实测 | 余额超 cap 时**请求不发** |
| **20 连续错就停** | API 不可用时早停 | 不让一次跑完烧光预算 |
| **key 仅 env** | 不落盘不 echo | `--key-env ''` 时连 Authorization 都不发 |

### 三个最常见的"看着不对"现象

1. **`schema_validity` 偏低**
   模型有时返了概率但 keys 不完整（漏 label）→ 该题记 invalid 并计 0。
   多数情况下：模型 rounding 把 9 项题舍入到 `0.999`，落在 2% 容忍带里被 renorm 救回；
   真正低的：模型 prompt 让它"用自然语言回答"了。

2. **`accuracy` 高但 `brier` / `ece` 也高**
   答对了但置信度没校准——自信错的也自信对。
   `native` 分布（TypeSafe 的 Jev、djev 等）通常校准好；`verbalized`（让通用 LLM 写概率）容易翻车。

3. **大量 error 记录**
   API 返了非 200 / 空 content。推理模型（gpt-5* / gpt-6*）若把输出预算全用在推理上会返回空 content；
   adapter 已把它们的输出上限提到 2048 并设 reasoning_effort="low"，仍空则调大 `max_output_tokens`。

### 预算 ledger 怎么看

`runs/notebook-demo/real/<provider>/ledger.jsonl` 是 reserve/settle 流水。
任何 unsettled 的 reservation 都计为已花——中断不退款。


# Cell 16 — 下一步 + CLI 速查

## 下一步

### 添加新 provider

只需在 `llm_eval/adapters/__init__.py` 的 `PROVIDER_TABLE` 里加一条（preset），并在 `llm_eval/adapters/` 加个 8 行 thin shim 继承 `OpenAICompatAdapter`。无需改 runner。

### 添加新 metric

在 `llm_eval/metrics.aggregate()` 加聚合逻辑；在 `llm_eval/summarize._PUBLIC_FIELDS` 加白名单键——两者缺一不可，否则公共导出里看不到。

### 添加新题族

在 `tasks/` 下放 JSONL，每行一个 task record（用 `llm_eval.task.Task` 的字段）。在 `scripts/convert_jevbench.py` 的 `_TOPIC_HINTS` 里给新 family 配 topic。

### CLI 速查

```
llm-eval run        跑单个 (adapter, model) 对一组题
llm-eval summarize   把 results.jsonl 汇总成 summary.json
llm-eval multi-run   一次配置多个 runner 并行（按 ≤5 一批分批）
llm-eval compare     跨 runner 对比表（markdown + json）
```

每条命令加 `--help` 看完整参数。

### 跑更多题

- `tasks/smoke.jsonl` — 6 题（秒级）
- `tasks/public/easy.jsonl` — 48 题（默认）
- `tasks/public_all.jsonl` — 231 题（≈$0.04-0.05/provider）

### 参考

- 完整文档：`README.md` 在 repo 根
- 实现细节：`llm_eval/IMPLEMENTATION.md`（设计纪律详解）
- 上游规范：[github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (MIT)

