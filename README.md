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

### 3. Using conda

```sh
conda env create
conda activate research-agent
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

### Direct Execution

Store output in a file:

```sh
uv run main.py 2>&1 | tee logs/output.log
```

### Running with Aspire (Recommended for Development)

Aspire provides a dashboard for monitoring logs, environment variables, and services.

**1. Install Aspire CLI:**

```sh
dotnet tool install -g Aspire.Cli
```

**2. Run with Aspire:**

```sh
aspire run
```

The Aspire dashboard will open automatically at `http://localhost:18888`, showing:
- Real-time logs from the agent
- Service status and resource usage
- Environment variables configuration
