# Contributing to Nodetool Core

Thank you for considering contributing to Nodetool Core! This document provides guidelines and instructions for contributing to the project.

## Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass (`make test`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Environment Setup

1. Clone your fork of the repository
2. Install development dependencies:

   ```bash
   make dev-install
   ```

   This will install the package in development mode and install all development dependencies.

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

This project uses:

- Black for code formatting
- Flake8 for linting
- MyPy for type checking

You can run all style checks with:

```bash
make lint
```

And format your code with:

```bash
make format
```

## Testing

Write tests for all new features and bug fixes. Run the tests with:

```bash
make test
```

For test coverage report:

```bash
make test-cov
```

## Documentation

Update documentation for any changes to the public API. Build the documentation with:

```bash
make docs
```

## Pull Request Process

1. Ensure your code passes all tests and style checks
2. Update documentation if needed
3. Update the CHANGELOG.md file with your changes
4. The PR should be reviewed by at least one maintainer
5. Once approved, a maintainer will merge your PR

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.
