# User memory

Store facts across sessions, search them, remove duplicates and resolve conflicts.
`memory_manager.py` implements notes, JSON and advanced JSON storage. The CLI
supports fully local operations; conversation extraction and agent conversations
use an LLM API.

From this directory:

```bash
python -m pip install -r requirements.txt
python memory_cli.py demo                  # temporary store; no API
python memory_cli.py --help
python memory_cli.py add --user alice --content 'Prefers tea' --tags drink --store-path /tmp/example-memories
python memory_cli.py query --user alice --query tea --store-path /tmp/example-memories
python memory_cli.py consolidate --user alice --store-path /tmp/example-memories
python memory_cli.py show --user alice --store-path /tmp/example-memories
```

Use an explicit `--store-path` to keep experiments separate from existing memory
files. `demo` uses a temporary directory by default.

For LLM-backed extraction, copy `env.example` to `.env`, configure the desired
supported provider, then run `python memory_cli.py extract --help`. `main.py`
provides interactive conversations; `background_memory_processor.py` processes
conversation history into memory updates. These paths can make paid API calls.

Run the local regression suite with `python -m pytest -q`, or from the parent
folder use `python run_checks.py --suite user-memory`.
