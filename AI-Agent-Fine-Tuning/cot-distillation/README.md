# CoT Distillation: Collecting SFT Data from Frontier Cloud Models

Companion code for **experiment 7-9 (chain-of-thought distillation)**. The first
step of SFT is obtaining high-quality demonstration data, and the most efficient
way to get it is to **distill a frontier model**: collect the teacher's
"thinking + answer" trajectories through bulk API calls, filter them with a
rule-based verifier, and use what survives as training data for the student.
(This is the route DeepSeek-R1 took when distilling smaller models.)

## Method

Three steps, corresponding to step one of experiment 7-9, "collect trajectories":

1. **Sample the task.** `problems.jsonl` ships 24 real AIME problems (1986-2024,
   stratified by problem number: 8 each from P1-5 / P6-10 / P11-15, with
   diagram-based problems removed). Answers are integers in 0-999, so a rule-based
   verifier can grade them automatically. For a low-cost smoke test, run a couple of
   problems with `--max_problems 2` rather than the full set.
2. **Collect trajectories.** `generate_data.py` calls the teacher through
   OpenRouter (`anthropic/claude-opus-4.8` by default) with the `reasoning`
   parameter enabled. Note that the Claude API returns **summarized thinking**,
   rewritten by a separate summarizer model — the token-by-token raw chain of
   thought only exists inside the encrypted `signature` field and is not exposed
   by the API — and newer models summarize more aggressively (see the measurements
   below). If you need the verbatim chain of thought, use an open model's native
   API instead, such as Kimi K3.
3. **Verify and filter.** The rule-based verifier checks the `Final Answer` value
   and keeps only correct trajectories, written as SFT data in the messages format
   `question -> <think>thinking</think> + final answer`. A student imitates a
   flawed thought process just as readily as a sound one, so this step is not
   optional.

## Choosing a teacher model: open-source SOTA by default

Most people doing post-training do **not** need to distill a closed model's chain
of thought. The gap between today's best open models (DeepSeek V4, Kimi K3,
GLM 5.2) and closed SOTA is smaller than it is usually assumed to be; if you are
post-training a model of 200B parameters or less, an open SOTA teacher is entirely
sufficient. The teacher only has to be *clearly better than the student*, not the
best model in the world.

The Claude runs are kept here as a controlled comparison: **how do a closed API's
summarized thinking and an open model's raw chain of thought actually differ as
SFT data?**

> Compliance note: this experiment only uses the reasoning/thinking capabilities
> exposed by each vendor's official API (Claude returns summarized thinking; open
> models such as Kimi K3 and DeepSeek return the raw chain of thought directly).
> It does not involve any means of bypassing a vendor's safety mechanisms. For
> closed models, use of the distilled output must follow the corresponding
> provider's terms.

## Run

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...

# Small smoke test (2 problems)
python generate_data.py --max_problems 2 \
  --sft_output /tmp/smoke_sft.jsonl --raw_output /tmp/smoke_raw.jsonl

# Full collection (24 AIME problems; Opus 4.8 emits ~40k output tokens, Kimi K3 ~60k)
python generate_data.py

# Dataset statistics
python analyze_data.py
```

Common arguments: `--model` to swap the teacher, `--base_url` / `--api_key_env` to
swap the endpoint, `--reasoning_effort` (for adaptive-thinking models such as
Opus 4.8) and `--reasoning_max_tokens` (for manual-budget models such as
Sonnet 4.5) to control the chain of thought, `--concurrency` for parallelism,
`--max_retries` for retries after a failure (each retry raises the temperature to
get a different trajectory), and `--request_timeout` for the per-request hard
timeout (essential when collecting from long-thinking models — see the
engineering lessons below). Run `python generate_data.py --help` for the full list.

## Output

| File | Contents |
| --- | --- |
| `data/sft_cot_distill_aime.jsonl` | Claude Opus 4.8 SFT training data (messages format, chain of thought wrapped in `<think>` tags) |
| `data/sft_cot_distill_aime_kimi_k3.jsonl` | Kimi K3 SFT training data |
| `data/raw_trajectories_*.jsonl` | Every raw trajectory, including those that failed verification, for analyzing the teacher's error patterns |

## AIME measurements: three teachers head to head (24 problems)

| | Claude Sonnet 4.5 | Claude Opus 4.8 | Kimi K3 |
| --- | --- | --- | --- |
| Verification pass rate | 22/24 | 24/24 | 23/24 |
| Nature of the chain of thought | Summary, near 1:1 fidelity | Summary, aggressively compressed | Raw chain of thought |
| Raw/visible token ratio (exact accounting) | 1.03-1.09 | **2.41** | **1.007** |
| Visible chain-of-thought size | median 6.2k chars | mean 536 tokens | mean 2.5k tokens |
| Problems with no chain of thought | 0 | 3 (skipped by adaptive thinking) | 1 (the abandoned timeout problem) |
| Trustworthiness of the `reasoning_tokens` field | understated to 55%-75% (OpenRouter side) | understated the same way | accurate (1.001) |

Token accounting method: use the model's own tokenizer (a `max_tokens=1` probe
reading `prompt_tokens`) to count the tokens in the visible chain of thought and
in the answer body; `completion_tokens - answer body tokens` is then the billed
raw thinking volume. The `reasoning_tokens` detail field returned by OpenRouter is
systematically understated for Claude and should not be taken at face value when
computing cost.

Three observations that bear directly on post-training:

1. **The newer the model, the higher the wall around its chain of thought.**
   Within the Claude family, Sonnet 4.5's summary is still close to verbatim,
   while Opus 4.8 compresses to under half and returns no chain of thought at all
   on 3 problems. Chain-of-thought transparency keeps tightening.
2. **Teacher capability is not the same as distillability.** Opus 4.8 has the
   highest accuracy yet yields the worst distillation material (sparse, truncated,
   missing summaries); Kimi K3 gets one fewer problem right, but every trajectory
   is the complete original text. Pick a teacher on both "can it solve this" and
   "will it show its work".
3. **Raw chains of thought carry meta-noise.** Kimi K3's transcripts contain
   English meta-reasoning, agonizing over output format (it once spent 700+ tokens
   on an easy problem debating whether to write `16` or `16%`), and mid-stream
   self-interruption. An answer verifier cannot filter noise like this, so a
   cleaning or rewriting pass is worth adding before using it for SFT.

Engineering lesson (a concrete instance of the book's "data pipeline robustness"):
Kimi K3 thought for over 15 minutes on individual AIME problems — `aime-2016-9-I`
exceeded 900 seconds on all three attempts and was ultimately abandoned — and the
Moonshot side can enter a half-open state where it stops sending but never closes
the connection. A collection pipeline must therefore flush each problem to disk as
it completes (this script writes `raw_trajectories` incrementally), impose a hard
timeout with `asyncio.wait_for` (an httpx read timeout does not help against a
half-open connection), and support retrying failures.

## Scaling up

To scale up, replace `problems.jsonl` with a problem source from your target
distribution (GSM8K or the MATH training set, for example), raise the concurrency,
and control coverage, diversity, and annotation accuracy according to the book's
"three dimensions of data quality".

---

## 中文

# CoT 蒸馏：从前沿云模型采集 SFT 数据

配套书中**实验 7-9（思维链蒸馏）**。SFT 的第一步是拿到高质量示范数据，而获取
SFT 数据最高效的方式就是**蒸馏前沿模型**：通过大规模 API 调用，把教师模型的
"思考 + 答案"轨迹采集下来，经规则验证器过滤后作为学生模型的训练数据
（DeepSeek-R1 蒸馏小模型走的就是这条路线）。

## 方法

三步流程（对应实验 7-9 的第一步"采集轨迹"）：

1. **采样任务**：`problems.jsonl` 内置 24 道 AIME 真题（1986–2024 年，按题号
   难度分层抽样：P1–5/P6–10/P11–15 各 8 道，已剔除含图形的题），答案是
   0–999 的整数，可以用规则验证器自动判对错。低成本冒烟测试用
   `--max_problems 2` 跑几道题即可，不必跑全量。
2. **采集轨迹**：`generate_data.py` 通过 OpenRouter 调用教师模型
   （默认 `anthropic/claude-opus-4.8`），开启 `reasoning` 参数获取思维链。
   注意：Claude API 返回的是 **summarized thinking**（由单独的摘要模型改写，
   逐 token 的原始思维链只存在于加密的 `signature` 字段中，API 不暴露），
   且模型越新摘要越激进（见文末实测）。若需要逐 token 原文，
   推荐直接用开放模型原生 API，例如 Kimi K3（见下文对照实验的运行参数）。
3. **验证过滤**：用规则验证器核对 `Final Answer` 数值，只保留答对的轨迹，
   写成 `问题 → <think>思考</think> + 最终答案` 的 messages 格式 SFT 数据。
   错误的思考过程会被学生一并模仿，所以这一步不能省。

## 教师模型怎么选：默认开源 SOTA，不必盯着闭源

对绝大多数做后训练的人来说，**不需要**去蒸馏闭源模型的思维链。当前最先进的
开源模型（DeepSeek V4、Kimi K3、GLM 5.2 等）与 SOTA 闭源模型的差距并没有
想象中大；如果你要后训练的是 200B 及以下规模的模型，用开源 SOTA 模型当教师
已经完全够用——教师的水平只需要"明显高于学生"，不需要"全球第一"。

本目录保留 Claude 的采集结果，目的是做一个对照：**闭源 API 的
summarized thinking 和开源模型的原始思维链，作为 SFT 数据到底有什么差别**。

> 合规说明：本实验只使用各厂商官方 API 提供的 reasoning/thinking 能力获取思维链
> （Claude 在 API 中返回 summarized thinking，Kimi K3、DeepSeek 等开放模型直接
> 返回原始思维链），不涉及任何绕过厂商安全机制的手段。对闭源模型，
> 蒸馏产物的使用需遵守对应服务商的条款。

## 运行

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...

# 小规模冒烟（2 道题）
python generate_data.py --max_problems 2 \
  --sft_output /tmp/smoke_sft.jsonl --raw_output /tmp/smoke_raw.jsonl

# 全量采集（24 道 AIME 题；Opus 4.8 输出约 4 万 token，Kimi K3 约 6 万）
python generate_data.py

# 数据统计
python analyze_data.py
```

常用参数：`--model` 换教师模型、`--base_url`/`--api_key_env` 换端点、
`--reasoning_effort`（Opus 4.8 等自适应思考模型）与 `--reasoning_max_tokens`
（Sonnet 4.5 等手动预算模型）控制思维链、`--concurrency` 并发数、
`--max_retries` 失败重试次数（重试时自动升温换取不同轨迹）、
`--request_timeout` 单请求硬超时（采集长思考模型时必备，见文末工程教训）。

## 输出

| 文件 | 内容 |
| --- | --- |
| `data/sft_cot_distill_aime.jsonl` | Claude Opus 4.8 的 SFT 训练数据（messages 格式，思维链包在 `<think>` 标签内） |
| `data/sft_cot_distill_aime_kimi_k3.jsonl` | Kimi K3 的 SFT 训练数据 |
| `data/raw_trajectories_*.jsonl` | 全部原始轨迹（含未通过验证的），用于分析教师错误模式 |

## AIME 实测：三位教师的对照（24 题）

| | Claude Sonnet 4.5 | Claude Opus 4.8 | Kimi K3 |
| --- | --- | --- | --- |
| 验证通过率 | 22/24 | 24/24 | 23/24 |
| 思维链性质 | 摘要，近 1:1 保真 | 摘要，激进压缩 | 原始思维链直出 |
| 原始/可见 token 比（精确对账） | 1.03–1.09 | **2.41** | **1.007** |
| 可见思维链规模 | 中位 6.2k 字符 | 均值 536 token | 均值 2.5k token |
| 无思维链的题 | 0 | 3（自适应思考跳过） | 0 |
| `reasoning_tokens` 字段可信度 | 虚低至 55%–75%（OpenRouter 侧） | 同样虚低 | 准确（1.001） |

token 对账方法：用模型自身 tokenizer（`max_tokens=1` 探针读 `prompt_tokens`）
数出可见思维链与正文的 token 数，`completion_tokens − 正文 token` 即为计费的
原始思考量。OpenRouter 返回的 `reasoning_tokens` 详情字段对 Claude 系统性虚低，
做成本核算时不可直接采信。

三个对后训练有直接意义的观察：

1. **模型越新，思维链围墙越高。** 同是 Claude，Sonnet 4.5 的摘要还接近逐字，
   Opus 4.8 已压到不足一半、且 3 道题完全不给思维链。思维链透明度在持续收紧。
2. **教师能力 ≠ 可蒸馏性。** Opus 4.8 答对率最高，给出的蒸馏材料却最差
   （摘要稀疏、截断、缺失）；Kimi K3 少对 1 题，但每条轨迹都是完整原文。
   选教师要同时看"会不会做"和"给不给看"。
3. **原始思维链含元噪声。** Kimi K3 的原文里有英文元思考、输出格式纠结
   （曾在简单题上用 700+ token 争论该写 `16` 还是 `16%`）、中途自我打断。
   答案验证器滤不掉这类噪声，用它做 SFT 前值得加一道清洗或重写。

工程教训（也是书中"数据管线健壮性"的实例）：Kimi K3 在个别 AIME 题上思考
超过 15 分钟（aime-2016-9-I 三次尝试均超 900 秒，最终放弃该题），且
Moonshot 端会出现"停止发送但不关闭连接"的半开状态。采集 pipeline 必须：
每题完成即落盘（本脚本增量写入 `raw_trajectories`）、用 `asyncio.wait_for`
做硬超时（httpx 读超时对半开连接无效）、失败可重试。

## 规模化

规模化的做法是把 `problems.jsonl` 换成目标分布的题目来源（如 GSM8K、MATH
训练集），提高并发，并按书中"数据质量三维度"控制覆盖面、多样性与标注准确性。
