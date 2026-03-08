# Contributing to Brain-Inspired AI

Thank you for your interest in contributing to Brain-Inspired AI! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/brain-inspired-ai.git
   cd brain-inspired-ai
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting
  ```bash
  black src/ tests/
  ```

- **Flake8**: Linting
  ```bash
  flake8 src/ tests/
  ```

- **MyPy**: Type checking
  ```bash
  mypy src/
  ```

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub

## Commit Message Format

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: add adaptive LIF neuron support

- Implement adaptive threshold mechanism
- Add tests for new neuron type
- Update documentation
```

## Areas for Contribution

### High Priority

- [ ] Improve STDP learning algorithms
- [ ] Optimize memory retrieval speed
- [ ] Add more neuron models (Izhikevich, AdEx)
- [ ] Implement distributed training
- [ ] Add more benchmarks

### Medium Priority

- [ ] Web interface improvements
- [ ] Additional tool integrations
- [ ] Documentation translations
- [ ] Example notebooks
- [ ] Performance optimizations

### Documentation

- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Tutorial videos
- [ ] Use case examples

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

## Code of Conduct

Be respectful and constructive in all interactions.
