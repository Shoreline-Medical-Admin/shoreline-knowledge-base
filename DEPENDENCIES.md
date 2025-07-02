# Dependency Management

This project uses both `pyproject.toml` (PEP 517/518 standard) and `requirements/*.txt` files for dependency management.

## Why Both?

1. **pyproject.toml**: Modern Python standard, used by pip, poetry, and other tools
2. **requirements.txt**: Legacy support, Docker compatibility, and deployment flexibility

## Installation Methods

### Using pyproject.toml (Recommended for Development)

```bash
# Install base package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with test dependencies
pip install -e ".[test]"

# Install with production dependencies
pip install -e ".[prod]"

# Install with all optional dependencies
pip install -e ".[dev,test,prod]"

# For LangGraph CLI support
pip install -e . "langgraph-cli[inmem]"
```

### Using requirements files (Alternative)

```bash
# Install base dependencies only
pip install -r requirements.txt

# Install for development
pip install -r requirements/dev.txt

# Install for testing
pip install -r requirements/test.txt

# Install for production
pip install -r requirements/prod.txt
```

## When to Use Which?

### Use pyproject.toml when:
- Setting up a development environment
- Building wheels/distributions
- Using modern Python tooling (poetry, hatch, etc.)
- Working with the package as a library

### Use requirements.txt when:
- Deploying with Docker
- Using legacy deployment systems
- Need exact version pinning
- Working with CI/CD systems that expect requirements.txt

## Keeping Them in Sync

When adding new dependencies:
1. Add to `pyproject.toml` in the appropriate section
2. Also add to the corresponding `requirements/*.txt` file
3. Run tests to ensure compatibility

## Core Dependencies

- **langgraph**: Core framework for building the agent
- **boto3**: AWS SDK for Python
- **langchain-aws**: LangChain AWS integrations
- **langchain-community**: Community LangChain components
- **python-dotenv**: Environment variable management

## Development Dependencies

- **pytest**: Testing framework
- **mypy**: Static type checking
- **ruff**: Fast Python linter
- **boto3-stubs**: Type hints for boto3

## Production Dependencies

- **gunicorn**: WSGI HTTP Server
- **uvicorn**: ASGI server for async support