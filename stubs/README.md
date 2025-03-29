# Type Stubs for Third-Party Packages

This directory contains type stub files (`.pyi`) for third-party packages that don't provide their own type hints. These stubs help Pylance and other type checkers understand the structure of external libraries without needing the actual implementation.

## Purpose

- Provide type information for libraries that don't include their own type hints
- Allow static type checking to work with optional dependencies
- Improve IDE experience with better autocomplete and error detection

## Included Stubs

- `py_ecc`: Type stubs for elliptic curve cryptography functions
- `spacy`: Type stubs for natural language processing functions

## How to Use

No action is needed to use these stubs - they're automatically configured in our project setup. The stub path is set in both:

- `.vscode/settings.json` for VSCode
- `pyrightconfig.json` for Pyright/Pylance
- `pyproject.toml` for Poetry users

## Adding New Stubs

If you need to add type stubs for another third-party package:

1. Create a directory structure that matches the package's import structure
2. Add `.pyi` files with type annotations for classes and functions
3. Include only the public interface (what users of the package would import)
4. Update `setup.py` to include the new stubs in the `data_files` section

### Example Structure for a New Library

```
stubs/
  ├── py.typed  # PEP 561 marker
  ├── some_library/
  │    ├── __init__.pyi  # Main package interface
  │    ├── module1.pyi   # Submodule definitions
  │    └── subpackage/
  │         └── __init__.pyi
```

### Example Stub File

```python
# stubs/some_library/__init__.pyi
from typing import List, Dict, Any, Optional

def main_function(param1: str, param2: Optional[int] = None) -> Dict[str, Any]: ...

class MainClass:
    attr1: str
    attr2: int
    
    def __init__(self, name: str) -> None: ...
    def method1(self, param: str) -> List[str]: ...
```

## Resources

- [PEP 561 - Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
- [mypy Documentation on Stubs](https://mypy.readthedocs.io/en/stable/stubs.html)
- [Pylance Type Checking Documentation](https://github.com/microsoft/pylance-release/blob/main/TYPING.md) 