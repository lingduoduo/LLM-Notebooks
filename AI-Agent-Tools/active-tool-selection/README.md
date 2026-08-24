# Active Tool Selection

An educational implementation of active tool discovery for LLM agents, inspired by the MCP-Zero paper ([arXiv:2506.01056](https://arxiv.org/pdf/2506.01056)).

### Overview

Traditional LLM agents inject all available tool schemas into prompts, creating massive context overhead and reducing agents to passive tool selectors. This project demonstrates **active tool discovery**, where agents autonomously identify capability gaps and request specific tools on-demand.

#### The Problem

Current tool integration approaches face critical limitations:

1. **Massive Context Overhead**: Injecting all tools can consume 100k+ tokens
2. **Passive Selection**: Agents select from pre-defined options rather than actively discovering
3. **Poor Scalability**: Context grows with ecosystem size, not task needs
4. **Lost Autonomy**: Tool selection delegated to external retrieval systems

#### The Solution: Active Tool Discovery

This project implements three core mechanisms from MCP-Zero:

1. **Active Tool Request**: Agents generate structured requests specifying their exact tool requirements
2. **Hierarchical Semantic Routing**: Two-stage matching algorithm (server-level → tool-level)
3. **Iterative Capability Extension**: Progressive toolchain building as task understanding evolves

### Strategy Comparison

This experiment turns tool selection into a measurable benchmark comparing three strategies on the **same task set**:

| Strategy | Description | Tools in context |
|------|------|----------------|
| `all-tools` | Inject all tools at once (traditional passive baseline) | All N tools |
| `retrieval` | Semantic top-k retrieval then inject (`RetrievalToolAgent`) | top-k only |
| `active` | MCP-Zero style active discovery: model iteratively requests tools (`ActiveToolAgent`) | grows on demand |

Entry point: `demo_comparison.py` with full argparse CLI:

```bash
# Offline only (deterministic, no API key): recall vs token cost vs scale
python demo_comparison.py --offline

# Pad catalog to 200 tools (synthetic distractors); watch token cost diverge
python demo_comparison.py --offline --num-tools 200

# End-to-end three-strategy compare (needs API key): correct tool calls, tokens, latency
python demo_comparison.py --strategy compare

# Single query, one strategy
python demo_comparison.py --query "Deploy version 2.0 to production" --strategy retrieval

# Save JSON results
python demo_comparison.py --offline --output results.json
```

Run `python demo_comparison.py --help` for all flags (`--strategy / --query / --num-tools /
--top-k / --model / --output / --offline / --legacy-demos`).

#### Offline benchmark (deterministic, no API)

`benchmark.py` provides a small labeled set (10 tasks, each with a ground-truth tool) and measures, **without any API call**:

- **Retrieval recall@k**: whether ground-truth tools land in the injected set
- **Schema tokens**: token cost of injected tool schemas (estimated from schema, deterministic)

Measured output of `python demo_comparison.py --offline` (top-k=5, 10 tasks):

| Strategy | Tools in context | Schema tokens | Recall (ground-truth reachable) |
|------|------|------|------|
| all-tools (full inject) | 35 | 3,857 | 100% |
| retrieval (top-5) | 4.7 average | 517 | 100% |

As the catalog grows, `all-tools` token cost scales linearly; `retrieval` stays flat (measured):

| Catalog size | all-tools tokens | retrieval(top-5) tokens | retrieval recall |
|------|------|------|------|
| 35 | 3,857 | 517 | 100% |
| 100 | 10,292 | 522 | 100% |
| 200 | 20,258 | 522 | 100% |
| 400 | 40,258 | 522 | 100% |

> Conclusion: retrieval-style on-demand selection keeps 100% recall while cutting schema tokens from thousands to hundreds, and does not inflate with ecosystem size—quantitative evidence for “turn tool selection into knowledge retrieval.” Numbers are deterministic from `--offline`.

#### End-to-end accuracy (needs API key)

With `OPENAI_API_KEY` set, `--strategy compare` actually calls the model and measures whether each strategy **invokes the ground-truth tool** (accuracy), plus mean tokens and latency. Online only—not part of the offline path.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Active Tool Agent                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Analyze Task & Identify Capability Gaps            │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  2. Generate Structured Tool Request                    │ │
│  │     <tool_request>                                      │ │
│  │       server: GitHub for repository operations          │ │
│  │       tool: search repositories by keyword              │ │
│  │     </tool_request>                                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Hierarchical Semantic Router                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Stage 1: Server-Level Routing                         │ │
│  │  • Match request to relevant servers/domains            │ │
│  │  • Filter by platform requirements                      │ │
│  │  • Return top-K servers                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Stage 2: Tool-Level Routing                           │ │
│  │  • Rank tools within selected servers                   │ │
│  │  • Semantic similarity matching                         │ │
│  │  • Return relevant tools                                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               Tool Knowledge Base                            │
│                                                              │
│  8 Servers × 35 Tools (optionally padded with distractors): │
│  • GitHub: Repository management (5 tools)                  │
│  • Filesystem: File operations (5 tools)                    │
│  • Database: SQL operations (5 tools)                       │
│  • Web: HTTP requests (4 tools)                             │
│  • Analytics: Data analysis (4 tools)                       │
│  • Communication: Email/messaging (4 tools)                 │
│  • DevOps: Deployment/monitoring (4 tools)                  │
│  • Cloud: Infrastructure management (4 tools)               │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### Core Files

- **`agent.py`**: Three agent implementations (one per strategy)
  - `ActiveToolAgent`: MCP-Zero style on-demand discovery (`active`)
  - `RetrievalToolAgent`: one-shot semantic retrieval of top-k tools (`retrieval`)
  - `PassiveToolAgent`: traditional approach with all tools pre-loaded (`all-tools`)

- **`benchmark.py`**: Labeled benchmark + offline evaluation
  - 10 tasks, each labeled with its ground-truth tool
  - `build_catalog(num_tools)`: real catalog, optionally padded with distractors
  - `evaluate_offline(...)`: deterministic recall@k / token-cost measurement (no API)

- **`tool_knowledge_base.py`**: Comprehensive tool catalog
  - 8 servers (domains) with 35 tools
  - Organized by platform/functionality
  - Simulates MCP ecosystem

- **`semantic_router.py`**: Hierarchical tool discovery
  - Two-stage semantic matching
  - TF-IDF based similarity
  - Structured request parsing

- **`config.py`**: Configuration settings
  - LLM provider settings
  - Routing thresholds
  - Agent parameters

#### Demo Scripts

- **`quickstart.py`**: Quick demonstration (⭐ Start here!)
- **`demo_comparison.py`**: Comprehensive comparison
- **`examples.py`**: Multiple use case examples

### Quick Start

#### 1. Install Dependencies

```bash
cd AI-Agent-Tools/active-tool-selection
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat

python -m pip install -r requirements.txt
```

#### 2. Configure API Key

```bash
cp env.example .env
# Edit .env and add your API key
```

> **Universal OpenRouter fallback**: if `OPENAI_API_KEY` is not set but
> `OPENROUTER_API_KEY` is, `config.py` automatically routes through OpenRouter
> (`base_url=https://openrouter.ai/api/v1`) and maps the model id to
> `provider/model` form (`gpt-*` → `openai/…`, `claude-*` →
> `anthropic/claude-opus-4.8`). Existing `OPENAI_BASE_URL`/`OPENAI_MODEL`

> overrides are preserved.

For direct Alibaba Cloud Model Studio / Bailian (Qwen), set
`LLM_PROVIDER=dashscope` (or `qwen`/`bailian`) and `DASHSCOPE_API_KEY`; the
default model is `qwen3.7-plus`. Set `DASHSCOPE_BASE_URL` for international keys.

#### 3. Run Quick Start

```bash
python quickstart.py
```

This will demonstrate:
- Active tool discovery process
- Passive tool injection (for comparison)
- Efficiency metrics and insights

#### 4. Run Tests

The automated tests are offline regressions and do not require an API key.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

### Performance Comparison

See the measured, reproducible numbers in **Strategy Comparison** above.
The offline table (schema-token cost + retrieval recall) is generated deterministically by
`python demo_comparison.py --offline` — no API key required. End-to-end accuracy/latency across
the three strategies requires an API key (`--strategy compare`).

> Note: earlier drafts of this README quoted round illustrative figures for token savings. Those
> have been replaced with the actual measured output of the offline benchmark to avoid fabricated
> numbers.

### Key Concepts

#### Active Tool Request

Instead of pre-loading tools, agents explicitly request what they need:

```python
<tool_request>
server: GitHub for repository management
tool: search repositories by stars and language
</tool_request>
```

#### Hierarchical Semantic Routing

Two-stage algorithm reduces search complexity:

1. **Stage 1**: Match to relevant servers (platforms)
   - "GitHub operations" → GitHub server
   - "File management" → Filesystem server

2. **Stage 2**: Match to specific tools within servers
   - "search repositories" → `github_search_repos`
   - "read file" → `fs_read_file`

#### Iterative Capability Extension

Tools are discovered progressively:

```
Turn 1: Need GitHub access
  → Load GitHub tools

Turn 2: Need to analyze downloaded files
  → Additionally load filesystem tools

Turn 3: Need to visualize results
  → Additionally load analytics tools
```

Toolchain grows with task complexity, not ecosystem size.

### Examples

#### Example 1: Simple Task

```python
from agent import ActiveToolAgent

agent = ActiveToolAgent()
result = agent.execute_task("Search for Python ML repositories on GitHub")

# Agent discovers and loads only GitHub tools
print(f"Tools loaded: {result['metrics']['tools_loaded']}")  # 2-3 tools
print(f"Tokens used: {result['metrics']['tokens_used']}")    # ~2,000
```

#### Example 2: Multi-Domain Task

```python
task = """
1. Query database for user data
2. Analyze with statistics
3. Create visualization
4. Email report to team
"""

result = agent.execute_task(task)

# Agent builds cross-domain toolchain:
# Database → Analytics → Communication
print(f"Tools: {result['tools_loaded']}")
```

#### Example 3: Comparison

```python
from agent import ActiveToolAgent, PassiveToolAgent

# Active approach
active = ActiveToolAgent()
active_result = active.execute_task(task)

# Passive approach
passive = PassiveToolAgent()
passive_result = passive.execute_task(task)

# Compare efficiency
reduction = (1 - active_result['metrics']['tokens_used'] /
             passive_result['metrics']['tokens_used']) * 100
print(f"Token reduction: {reduction:.1f}%")  # Typically 90-98%
```

### Running Demonstrations

#### 1. Quick Start (Recommended first)

```bash
python quickstart.py
```

Shows basic active vs passive comparison.

#### 2. Strategy Comparison Benchmark (main experiment)

```bash
python demo_comparison.py --offline   # deterministic, no API key needed
python demo_comparison.py             # + end-to-end accuracy/latency if API key present
```

By default it prints:
- Offline strategy table: retrieval recall@k vs. tool-schema token cost (deterministic)
- Scaling table: token cost as the catalog grows to hundreds of tools
- End-to-end table (with API key): accuracy / tokens / latency for `all-tools`, `retrieval`, `active`

The original narrative demos (semantic-routing walk-through, iterative discovery, etc.) are still
available via `--legacy-demos`. See the Strategy Comparison section and `python demo_comparison.py --help` for all flags.

#### 3. Use Case Examples

```bash
python examples.py
```

Demonstrates:
- GitHub workflow
- Data pipeline
- DevOps automation
- Multi-turn discovery
- Efficiency metrics

### Use Cases

#### Ideal for Active Discovery

1. **Task-Specific Operations**: Known scope, specific tools needed
2. **Large Tool Ecosystems**: 50+ tools where most are irrelevant
3. **Multi-Turn Conversations**: Tools needed evolve over time
4. **Token-Constrained Environments**: Limited context windows

#### When Passive Might Work

1. **Small Tool Sets**: <10 tools total
2. **All Tools Relevant**: Every tool likely to be used
3. **Single-Turn Tasks**: No iterative refinement

### Configuration

Edit `config.py` or set environment variables:

```python
# LLM Configuration
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5.6-luna"

# Routing Configuration
SIMILARITY_THRESHOLD = 0.15  # Min similarity for tool match
TOP_K_SERVERS = 3            # Number of servers to search
TOP_K_TOOLS = 5              # Tools to return per server

# Agent Configuration
MAX_TOOL_REQUESTS = 5       # Max discovery iterations
```

### Educational Value

This project demonstrates:

#### Software Engineering Principles

- **Separation of Concerns**: Router, knowledge base, agent cleanly separated
- **Scalability**: Efficient with 10 or 1,000 tools
- **Modularity**: Easy to add new servers/tools

#### AI Agent Design Patterns

- **Active vs Passive**: Fundamental architectural difference
- **Hierarchical Search**: Reduces complexity from O(n) to O(log n)
- **Semantic Matching**: Beyond keyword matching

#### Real-World Applications

- **Tool Ecosystems**: MCP, LangChain, AutoGen
- **Agent Frameworks**: Building production-ready agents
- **Context Management**: Handling long contexts efficiently

### Technical Details

#### Semantic Routing Implementation

Uses TF-IDF vectorization with cosine similarity:

```python
# Server-level
server_vector = vectorizer.transform([request])
similarities = cosine_similarity(server_vector, server_embeddings)

# Tool-level
tool_vector = vectorizer.transform([request])
tool_similarities = cosine_similarity(tool_vector, tool_embeddings)

# Combined score
final_score = 0.3 * server_score + 0.7 * tool_score
```

#### Token Estimation

Approximates tokens in tool schemas:

```python
def count_tokens_in_schema(schema):
    # Rough estimation: 1 token ≈ 4 characters
    schema_str = json.dumps(schema)
    return len(schema_str) // 4
```

### Key Insights

1. **Autonomy Matters**: Agents should control their capability acquisition
2. **Context is Expensive**: Every token counts at scale
3. **Semantic Matching Works**: TF-IDF sufficient for tool discovery
4. **Iteration Enables Flexibility**: Static tool sets can't anticipate needs
5. **Hierarchical Search Scales**: Two-stage routing maintains performance

### References

- **MCP-Zero Paper**: [arXiv:2506.01056](https://arxiv.org/pdf/2506.01056)
- **Model Context Protocol**: [Official Repository](https://github.com/modelcontextprotocol)
- **Tool Learning Survey**: [ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3708498)

### Extending the Project

#### Add New Tools

```python
# In tool_knowledge_base.py
new_tool = ToolDefinition(
    name="your_tool_name",
    description="What the tool does",
    parameters={...},
    server="server_name"
)
```

#### Add New Server

```python
# Create tools for the server
tools = [...]

# Add server
servers.append(ServerDefinition(
    name="your_server",
    description="Server description",
    tools=tools
))
```

#### Customize Routing

```python
# In config.py
SIMILARITY_THRESHOLD = 0.5  # More strict matching
TOP_K_SERVERS = 5           # Search more servers
```

### Troubleshooting

#### API Key Issues

```bash
# Verify that the .env file exists without printing its secrets
test -f .env && echo ".env is present"
```

#### Import Errors

```bash
# Reinstall this experiment's dependencies
python -m pip install --upgrade -r requirements.txt
```

#### Low Similarity Scores

```bash
# Lower threshold in config.py
SIMILARITY_THRESHOLD = 0.2
```

### Future Enhancements

Potential improvements:

1. **Better Embeddings**: Use sentence-transformers or OpenAI embeddings
2. **Caching**: Cache tool embeddings for faster routing
3. **Feedback Loop**: Learn from tool usage patterns
4. **Multi-Agent**: Tool sharing between agent instances
5. **Real Tools**: Connect to actual APIs instead of simulation

### License

MIT License - See LICENSE file for details

### Acknowledgments

- Inspired by MCP-Zero paper by Xiang Fei, Xiawu Zheng, and Hao Feng
- Based on Model Context Protocol (MCP) ecosystem
- Built for educational purposes in AI Agent development

**Ready to get started?** Run `python quickstart.py` to see active tool discovery in action!
