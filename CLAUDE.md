# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARIEL (Autonomous Robots through Integrated Evolution and Learning) is a Python framework for evolutionary robotics research. It combines evolutionary computation with MuJoCo physics simulation to evolve both robot bodies (morphologies) and brains (controllers).

## Commands

### Setup
```bash
uv venv && uv sync
```

### Run examples
```bash
uv run examples/re_book/1_brain_evolution.py
```

### Testing
```bash
nox                          # Full test suite (Python 3.12 + 3.13)
nox --session=tests          # Unit tests only
```

### Linting and formatting
```bash
ruff check --fix             # Lint with auto-fix
ruff format                  # Format code
```

### Type checking
```bash
mypy src/ariel/              # Strict mode, Python 3.12
pyright                      # Also strict mode
```

### Documentation
```bash
nox --session=docs           # Build + serve with live reload
```

### Pre-commit hooks
```bash
nox --session=pre-commit -- install   # Install hooks
```

Hooks run: ruff check, ruff format, pydoclint, prettier, trailing whitespace fix, end-of-file fix, large file check.

## Architecture

Source code lives in `src/ariel/` with these core modules:

- **`ec/`** — Evolutionary computation engine. `ea.py` is the main EA class with SQLite persistence. `individual.py` uses SQLModel for database-backed individuals with JSON-serializable genotypes. `population.py` provides a chainable query API. `genotypes/` contains tree-based (NetworkX), NDE (PyTorch), and CPPN genome representations.

- **`body_phenotypes/`** — Robot morphology. `robogen_lite/` implements a modular robot system where bodies are graphs of Module subclasses (Core, Brick, Hinge). `constructor.py` converts these graphs to MuJoCo specs. `lynx_mjspec/` has Lynx arm specifications.

- **`simulation/`** — Physics simulation layer. `environments/` has terrain generators inheriting from `BaseWorld` (MuJoCo MjSpec-based). `controllers/` implements CPG-based control systems via a callback-based `Controller` class. `tasks/` defines evaluation tasks (locomotion, turning, gait learning).

- **`parameters/`** — Configuration via Pydantic BaseSettings (`ArielConfig`) and dataclasses. Type aliases in `ariel_types.py`.

- **`utils/`** — Simulation runners, video recording, morphological descriptors, MuJoCo helpers, noise generation.

- **`visualisation/`** — Plotly/Dash dashboard and analysis tools.

### Key design patterns

- Genotypes (EC) are separated from phenotypes (body_phenotypes), connected by decoders
- `BaseWorld` → `MjSpec` → `MjModel` pipeline for physics environments
- Controllers use callbacks to decouple control logic from the simulation loop
- Robot morphologies are NetworkX DiGraphs of Module nodes

## Code Style

- **Line length:** 80 characters
- **Quotes:** double quotes
- **Docstrings:** NumpyDoc style (enforced by pydoclint)
- **Type hints:** full strict typing (mypy + pyright)
- **Linting:** `ruff` with `select = ["ALL"]` — nearly all rules enabled
- **Import order:** standard library, then third-party, then local
- **Coverage:** 100% required (enforced in pyproject.toml)
- **Python:** 3.12+ (targets 3.12 and 3.13)

## Key Dependencies

MuJoCo (physics), DEAP/EvoTorch/Nevergrad (evolution), PyTorch (neural networks), Pydantic (config), SQLModel (persistence), NetworkX (graphs), Plotly (visualization).
