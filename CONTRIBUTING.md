# Contributing to Cortiqa Python SDK

Thank you for your interest in contributing to the **Cortiqa Python SDK**! We welcome bug fixes, documentation improvements, feature additions, and community feedback.

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cortiqa-ai/cortiqa-sdk-python.git
   cd cortiqa-sdk-python
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install the package in editable mode:**
   ```bash
   pip install -e ".[dev]"
   ```

---

## Running Tests

Run the test suite using Python's test runner:

```bash
# Using pytest
pytest -v

# Or using standard unittest
python -m unittest discover tests
```

---

## Code Quality & Formatting

We follow standard PEP 8 conventions:

* Format code using `black`:
  ```bash
  black cortiqa tests examples
  ```
* Check typing using `mypy`:
  ```bash
  mypy cortiqa
  ```

---

## Submitting Pull Requests

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. Commit your changes with clear, semantic commit messages (e.g., `feat:`, `fix:`, `docs:`).
3. Ensure all existing and new unit tests pass.
4. Open a Pull Request on GitHub describing your changes.

---

## License

By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
