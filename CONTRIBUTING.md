# Contributing to Mistral Inference

Thank you for your interest in contributing to Mistral Inference! This guide will help you get started.

## Prerequisites

- **Python** >= 3.9.10
- **Poetry** for dependency management
- **GPU with CUDA support** (required by xformers)

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/mistral-inference.git
cd mistral-inference
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Create a Branch

```bash
git checkout -b <type>/<description>
# Examples: fix/nccl-broadcast, docs/update-readme, feat/new-model
```

## Development Workflow

### Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
poetry run ruff check src/ tests/

# Auto-fix issues
poetry run ruff check --fix src/ tests/

# Format code
poetry run ruff format src/ tests/
```

**Configuration**: Line length is 120 characters (see `pyproject.toml`).

### Type Checking

[Mypy](https://mypy-lang.org/) is configured in strict mode:

```bash
poetry run mypy src/
```

All functions must have type annotations (`disallow_untyped_defs = true`).

### Running Tests

```bash
poetry run pytest tests/
```

> **Note**: Some tests require GPU access. If you don't have a GPU available, please mention this in your PR and the maintainers can verify on their infrastructure.

## Submitting a Pull Request

1. Ensure your code passes all checks locally:
   ```bash
   poetry run ruff check src/ tests/
   poetry run ruff format --check src/ tests/
   poetry run mypy src/
   poetry run pytest tests/
   ```

2. Write a clear PR title following the pattern: `type(scope): description`
   - `fix(cache): resolve device mismatch on multi-GPU`
   - `docs(readme): update deployment instructions`
   - `feat(model): add support for new architecture`

3. In the PR description:
   - Explain **what** the change does and **why**
   - Reference related issues (e.g., `Fixes #123`)
   - Note any breaking changes

4. Keep PRs focused — one logical change per PR

## Types of Contributions

We welcome:
- **Bug fixes** — especially for multi-GPU and distributed inference
- **Documentation** — README improvements, docstrings, examples
- **Tests** — expanding test coverage
- **Performance** — optimizations with benchmarks showing improvement
- **Tutorials** — new examples in the `tutorials/` directory

For larger changes (new model architectures, significant refactors), please open an issue first to discuss the approach.

## Reporting Issues

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) to file issues. Include:
- Your environment (Python version, GPU, OS)
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback

## Code of Conduct

Be respectful and constructive in all interactions. We're building open-source AI infrastructure together.
