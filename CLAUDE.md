# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[verl]      # Verl backend (GPU, distributed training)
uv pip install -e .[tinker]    # Tinker backend (single-machine, CPU-friendly)
uv pip install -e .[all]       # All optional dependencies
```

Python >= 3.11 required. verl and tinker backends are mutually exclusive; tinker is for single-machine setups, verl for multi-GPU.

### Linting & Formatting
```bash
pre-commit run --all-files     # Run ruff lint (includes formatting)
```

Ruff config (`pyproject.toml`): selects E, F, UP, B, I, G rules; ignores F405, F403, E731, B007, UP032, G004, E712. Line length is 200.

### Testing
```bash
pytest tests/                            # All tests
pytest tests/agents/test_agent.py       # Single test file
pytest tests/agents/test_agent.py::test_base_agent  # Single test
```

### Type Checking
```bash
mypy rllm/
```
Excludes: `verl/.*`, `rllm/rewards/code_utils/.*`, `rllm/rewards/math_utils/.*`, `examples/`, `scripts/`, `tests/`, `docs/`.

### Documentation
```bash
./build_docs.sh          # Build docs
./build_docs.sh serve    # Build and serve at http://localhost:8000
```

### CLI
```bash
rllm --help              # Entrypoint: rllm.experimental.cli.main:cli
rllm train --help
rllm eval --help
rllm init --help
```

## Architecture

rLLM is a framework for post-training language agents via reinforcement learning. The core abstraction layers are:

### Core Interfaces

**`rllm/agents/agent.py`** — `BaseAgent` abstract class: `act(observation) → Action`, `reset()`. Agents are stateful classes that produce actions from observations.

**`rllm/environments/base/base_env.py`** — `BaseEnv` abstract class: `reset() → (obs, info)`, `step(action) → (obs, reward, done, info)`. Must implement `idx` property (batch identifier).

**`rllm/types.py`** — Core Pydantic models: `Step` (single LLM interaction), `Trajectory` (sequence of Steps), `Episode` (rollout unit containing trajectories). Extended training variants live in `rllm/agents/agent.py`.

**`rllm/workflows/workflow.py`** — `Workflow` abstract class: `async run(task, uid, **kwargs) → Episode`. Composes agent + environment into a full episode.

### Execution Engines (`rllm/engine/`)

- **`AgentExecutionEngine`**: Orchestrates parallel agent-environment interactions. Key params: `n_parallel_agents` (default 128), `max_steps`, `max_response_length`, `gamma`. Computes MC returns and advantages.
- **`AgentWorkflowEngine`**: Episode-level orchestration using Workflow objects.
- **`AgentSDKEngine`**: SDK-based trace collection for data generation.

### Training System (`rllm/trainer/`)

**High-level API** (recommended): Use `AgentTrainer` from `rllm.experimental.unified_trainer` with the `@rllm.rollout` and `@rllm.evaluator` decorators:

```python
from rllm.experimental.unified_trainer import AgentTrainer

trainer = AgentTrainer(
    backend="tinker",  # or "verl" for GPU
    agent_flow=solve,  # @rllm.rollout decorated function
    evaluator=score,   # @rllm.evaluator decorated function
    config=config,
    train_dataset=dataset,
)
trainer.train()
```

**Backends**: `verl` (Ray + vLLM, multi-GPU), `tinker` (single-machine, Python 3.11+), `fireworks` (cloud).

**`rllm/experimental/unified_trainer.py`** — Backend-agnostic async trainer. Handles episode collection, advantage computation, rejection sampling, compact filtering, distillation.

Configuration is via Hydra DictConfig. Base configs live in `rllm/trainer/config/` (e.g., `agent_ppo_trainer.yaml`). Key config namespaces: `rllm.agent.*`, `rllm.env.*`, `rllm.workflow.*`, `data.*`, `model.*`, `trainer.*`, `actor_rollout_ref.*`.

### Training Data Flow

```
AgentTrainer.fit()
  → UnifiedTrainer (async)
      → RolloutEngine → AgentExecutionEngine/AgentWorkflowEngine
          → parallel (Agent ↔ Env) interactions → Episodes
      → compute rewards, advantages, filtering
      → BackendProtocol.train_step(batch)
```

### SDK (`rllm/sdk/`)

Trace collection system for generating training data. Key APIs:
- `session()`: Context manager to capture LLM calls
- `@trajectory`: Decorator that auto-collects calls into a Trajectory
- `get_chat_client()` / `get_chat_client_async()`: Wrapped OpenAI clients
- Backends: in-process `ContextVarSession` or distributed `OpenTelemetrySession`
- Storage: `SqliteTracer` (persistent) or `InMemorySessionTracer`

### Rewards (`rllm/rewards/`)

`MathReward`, `CodeReward`, `SearchReward`, `CountdownReward`. Optional deps: sympy, antlr4, pylatexenc for symbolic math.

### Tools (`rllm/tools/`)

Central `ToolRegistry` with built-in: `PythonInterpreter`, `GoogleSearchTool`, `TavilySearchTool`, `FirecrawlTool`. MCP integration available via `[tools]` extra.

### Lazy Loading

`rllm/__init__.py` uses `__getattr__` for lazy imports to minimize startup overhead. Extend with care.

## Key Conventions

- Agent, environment, workflow, tool, and reward implementations are registered in `rllm/registry/` JSON files for CLI discovery.
- Experimental/next-gen code lives under `rllm/experimental/` — this is the actively developed path.
- `rllm/patches/` contains monkey patches for third-party libraries (apply carefully).
- Workflows should extend `Workflow` and implement `async run()`. Use `postprocess_episode()` for consistent error/termination handling.
- Pydantic `BaseModel` constructors must use keyword arguments.
- **Prefer the high-level `@rllm.rollout` / `@rllm.evaluator` + `AgentTrainer` API** for training workflows. The BaseAgent/BaseEnv/Workflow abstractions are lower-level building blocks.

## Additional CLI Commands
```bash
rllm model setup         # Configure model provider
rllm dataset list        # List available datasets
rllm init                # Initialize a new rllm project
rllm login               # Login to model providers
```

## Directory Structure

- **`agenthub/`** — Framework-specific agent implementations (SmolAgent, Strands, LangGraph, terminal, SWE agents). Install via `[all]` or individually.
- **`rllm-model-gateway/`** — LiteLLM-based proxy that captures token IDs and logprobs for training. Acts as the inference layer between agents and model providers. Used automatically when training — agents point to the gateway URL.
- **`rllm/experimental/fully_async/`** — Fully async PPO training with decoupled rollout and training via message queue. Uses SGLang backend for rollout generation.

## Datasets & Benchmarks

The CLI ships with 50+ benchmarks defined in `rllm/registry/datasets.json`. Categories: `math`, `mcq`, `code`, `vlm`, `search`, `agentic`, `qa`, `translation`, `instruction_following`. Run `rllm eval <dataset>` to evaluate, `rllm train <dataset>` to train.

## RL Algorithms

Supported algorithms via `rllm/experimental/common/rl_algo.py`: GRPO, REINFORCE, RLOO, rejection sampling. Configure via `trainer.rl_algo` in Hydra configs.
