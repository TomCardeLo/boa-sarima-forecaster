# Contributing to boa-forecaster

Thanks for your interest in contributing!

## Getting started

1. Fork the repo and clone your fork
2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,ml]"
   ```
3. Create a feature branch: `git checkout -b feature/my-feature`

## Development workflow

- Run tests before submitting: `pytest tests/ -v`
- Run linting: `ruff check src/`
- Write tests for new features in `tests/unit/` or `tests/integration/`
- Follow existing code style and patterns

## Adding a new model

Implement the `ModelSpec` protocol (see `src/boa_forecaster/models/base.py`) and register it:

```python
from boa_forecaster.models import register_model
register_model("my_model", MyModelSpec)
```

See any existing model in `src/boa_forecaster/models/` as reference.

## Pull requests

- Keep PRs focused on a single change
- Update tests and docs if applicable
- Target the `main` branch
- Describe what and why in the PR description

## Reporting issues

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
