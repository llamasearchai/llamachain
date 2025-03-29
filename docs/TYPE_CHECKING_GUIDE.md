# Type Checking Guide for LlamaChain

This guide explains how to handle type checking with optional dependencies in the LlamaChain codebase.

## Using TYPE_CHECKING for Optional Dependencies

When working with optional dependencies, we use the `TYPE_CHECKING` constant from the `typing` module. This allows us to have import statements that are only processed by type checkers (like Pylance or mypy) but not at runtime.

### Basic Pattern

```python
from typing import TYPE_CHECKING, Optional, List

# Import for type checking only
if TYPE_CHECKING:
    from optional_dependency import SomeClass
    from another_optional_dependency import AnotherClass

# Runtime imports
try:
    from optional_dependency import SomeClass
    OPTIONAL_DEPENDENCY_AVAILABLE = True
except ImportError:
    OPTIONAL_DEPENDENCY_AVAILABLE = False

class MyClass:
    def __init__(self):
        if OPTIONAL_DEPENDENCY_AVAILABLE:
            self.obj = SomeClass()
        else:
            self.obj = None
    
    def method_using_optional_import(self) -> Optional["AnotherClass"]:
        if not OPTIONAL_DEPENDENCY_AVAILABLE:
            return None
        # Use the optional dependency
        # ...
```

## Examples in LlamaChain Codebase

### Example 1: py_ecc in verifier.py

```python
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING, Tuple, cast

# Import for type checking only
if TYPE_CHECKING:
    from py_ecc.bn128 import (
        G1, G2, 
        add, curve_order, multiply, neg, pairing
    )

class ZKVerifier:
    def __init__(self):
        try:
            # Use explicit relative imports to help Pylance locate the module
            import py_ecc.bn128
            self._bn128 = py_ecc.bn128
            self.py_ecc_available = True
        except ImportError as e:
            self._bn128 = None
            self.py_ecc_available = False
```

### Example 2: spaCy in NLP modules

```python
from typing import List, Dict, Optional, TYPE_CHECKING

# Import for type checking only
if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc, Token, Span

# Runtime imports
try:
    import spacy
    from spacy.tokens import Doc, Token, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class EntityExtractor:
    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name
        self.nlp = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                # Model not found
                self.nlp = None
```

## Type Comments for Special Cases

Sometimes you may need to use inline type comments to make Pylance happy:

```python
# Option 1: Ignore the specific import
import special_lib  # type: ignore

# Option 2: Ignore a particular error on a line
result = special_function()  # type: ignore[attr-defined]

# Option 3: Cast the value for type checkers
from typing import cast
result = cast(int, some_value_that_pylance_thinks_is_wrong_type)
```

## Best Practices

1. **Always use TYPE_CHECKING pattern** for optional dependencies.
2. **Check availability before use** with a boolean flag like `SPACY_AVAILABLE`.
3. **Provide graceful fallbacks** when dependencies are not available.
4. **Use clear error messages** to help users understand why functionality might be limited.
5. **Use type stubs** (`*.pyi` files) for third-party libraries when appropriate.
6. **Add inline `# type: ignore` comments** only as a last resort.

## Tips for Common Libraries

### spaCy

```python
if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc
else:
    try:
        import spacy
        from spacy.tokens import Doc
        SPACY_AVAILABLE = True
    except ImportError:
        SPACY_AVAILABLE = False
        
# Always check before using
if SPACY_AVAILABLE:
    # Use spaCy
```

### Cryptographic Libraries (py_ecc, etc.)

```python
if TYPE_CHECKING:
    from py_ecc.bn128 import G1, G2, pairing
    
# Store imported module
try:
    import py_ecc.bn128 as bn128
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    bn128 = None  # Set to None for easy checks later
    
# Use with safeguards
if CRYPTO_AVAILABLE and bn128 is not None:
    # Use bn128
```

## Further Reading

- [Python typing module documentation](https://docs.python.org/3/library/typing.html)
- [mypy documentation on TYPE_CHECKING](https://mypy.readthedocs.io/en/stable/common_issues.html#import-cycles)
- [Pylance and pyright documentation](https://github.com/microsoft/pyright/blob/main/docs/type-checking.md) 