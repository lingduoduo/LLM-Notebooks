# models/

Laya 权重（约 2.2 GB）**不入库**，本目录只保留推理所需的两个上游模块：

| 文件 | 来源 | 说明 |
|---|---|---|
| `rl_agent_api.py` | ModelScope `convaiinnovations/laya` | `RLAgent`：加载 checkpoint 并执行 `system_one(state, questions)` |
| `rl_common.py` | 同上 | 模型构建、序列拼接、置信度与温度等公共实现 |

`client.py` 通过把本目录加入 `sys.path` 来 `from rl_agent_api import RLAgent`，因此这两个文件必须留在
`models/` 下，checkpoint 目录（含 `model.safetensors` / `rl_agent_config.json` / `encoder/` /
`tokenizer/`）也放在这里。

## 从 Hugging Face 下载权重

从仓库根目录运行（macOS / Linux）：

```bash
python3 -m venv Jev_LocalModel/.venv
source Jev_LocalModel/.venv/bin/activate
python -m pip install -U pip huggingface_hub
python -m pip install -r Jev_LocalModel/requirements.txt
hf download convaiinnovations/laya \
  --include "multilingual/**" \
  --local-dir Jev_LocalModel/models
```

模型仓库：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya/tree/main)。
默认只下载多语言 checkpoint，包括权重、模型配置和 tokenizer。下载完成后应包含：

```text
Jev_LocalModel/models/multilingual/
├── model.safetensors
├── rl_agent_config.json
├── encoder/config.json
└── tokenizer/
```

启动与验证：

```bash
python -m Jev_LocalModel.serve \
  --model-dir Jev_LocalModel/models/multilingual \
  --host 127.0.0.1 --port 8811
```

浏览器打开 <http://127.0.0.1:8811/> 查看说明，或打开 `/healthz` 检查加载状态。
如果报 `No model.safetensors`，请确认下载成功且 `--model-dir` 指向包含该文件的目录。
若要下载全部 checkpoint，去掉 `--include "multilingual/**"`；服务默认仍使用多语言版。

## 校验

权重可用 本仓库记录的 SHA256 核对（截至 2026-09-23）：

| 文件 | SHA256 |
|---|---|
| `model.safetensors` | `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` |
| `multilingual/model.safetensors` | `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` |
| `typed-decisions/model.safetensors` | `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e` |

```bash
shasum -a 256 Jev_LocalModel/models/multilingual/model.safetensors
```

下载 CLI 说明见 [Hugging Face 官方文档](https://huggingface.co/docs/huggingface_hub/guides/cli)。

## 许可

权重与上述两个模块来自 Convai Innovations 的 Laya 项目，Apache-2.0。本目录只做搬运与说明，
未修改上游代码；修改项仅在本仓库 `Jev_LocalModel/` 下 `models/` 外的 `client.py`（Jev 形状的调用封装）与 `serve.py`
（HTTP 服务）中，见上级目录 README。
