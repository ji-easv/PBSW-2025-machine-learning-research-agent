# Research paper AI Agents

The agent can answer the following prompt:

> Find a research paper on [topic] that was published [in/before/after] [year] and has [number of citations] citations.

## Getting Started (Environment Setup)

You can set up and run this project using your preferred Python environment manager:

### 1. Using UV

```sh
uv venv .venv
uv sync
source .venv/bin/activate
```

### 2. Using pip/venv

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

**To generate `requirements.txt` from `pyproject.toml` (using pip-tools):**

```sh
pip install pip-tools
pip-compile pyproject.toml --output-file requirements.txt
```

Or with UV:

```sh
uv pip freeze > requirements.txt
```

## Running the project

Run with UV:

```sh
uv run main.py
```

Run with a specific LLM provider:

```sh
uv run main.py --google
```

Run with Python:

```sh
python main.py
```

Allows you to store output in a file:

```sh
uv run main.py 2>&1 | tee logs/output.log
```
