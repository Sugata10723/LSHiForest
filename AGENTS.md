## Build, Lint, and Test Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Linting**: 
  - `flake8 .`
  - `black --check .`
- **Testing**: Tests are run in Jupyter Notebooks (`test.ipynb`). For a more structured approach, use `pytest`.
  - To run a single test file: `pytest LSHiForest/test_iforest.py` (assuming a test file is created).

## Code Style Guidelines

- **Formatting**: Use `black` for code formatting. 4-space indentation.
- **Imports**: Group imports: 1. standard library, 2. third-party, 3. local application. Sort them alphabetically.
- **Naming**:
  - `CamelCase` for classes (e.g., `LSHiForest`).
  - `snake_case` for functions, methods, and variables (e.g., `decision_function`).
- **Types**: Add type hints to function signatures where possible.
- **Docstrings**: Use Google-style docstrings for all public modules, classes, and functions.
- **Error Handling**: Raise specific exceptions (`ValueError`, `TypeError`) instead of generic `Exception`.
- **Comments**: Use comments to explain complex logic. The code has Japanese comments; continue to use them for consistency.
