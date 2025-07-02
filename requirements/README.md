# Requirements Structure

This directory contains modular requirements files for different environments and use cases.

## Files Overview

- **base.txt**: Core dependencies required for the application to run
- **dev.txt**: Development dependencies including linting, formatting, and development tools
- **prod.txt**: Production-specific dependencies for deployment
- **test.txt**: Testing frameworks and utilities
- **constraints.txt**: Version constraints to ensure compatibility across environments

## Usage

### For Development
```bash
pip install -r requirements/dev.txt
```

### For Production
```bash
pip install -r requirements/prod.txt
```

### For Testing Only
```bash
pip install -r requirements/test.txt
```

### With Version Constraints
```bash
pip install -c requirements/constraints.txt -r requirements/dev.txt
```

## Updating Dependencies

1. Add new dependencies to the appropriate file:
   - Core app dependencies → `base.txt`
   - Development tools → `dev.txt`
   - Testing tools → `test.txt`
   - Production servers → `prod.txt`

2. Update version constraints in `constraints.txt` for critical dependencies

3. Test the installation in a clean environment:
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # On Windows: test_env\Scripts\activate
   pip install -r requirements/dev.txt
   ```

## Dependency Tree

```
requirements.txt → base.txt
dev.txt → base.txt + development tools
prod.txt → base.txt + production servers
test.txt → base.txt + testing frameworks
```

## Best Practices

1. Keep base.txt minimal - only include dependencies needed for the app to run
2. Pin major versions in constraints.txt to avoid breaking changes
3. Regularly update dependencies for security patches
4. Test dependency updates in development before production
5. Use virtual environments to isolate dependencies