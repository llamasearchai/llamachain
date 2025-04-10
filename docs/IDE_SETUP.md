# VSCode and Pylance Setup Guide

This guide will help you set up your development environment with VSCode and Pylance for working with the LlamaChain project.

## Prerequisites

- [VSCode](https://code.visualstudio.com/) installed
- [Python Extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-python.python) installed

## Setting Up the Environment

### 1. Install Required Dependencies

First, install all the required packages in your virtual environment:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Additional Development Tools

```bash
# Install development tools
pip install -e ".[dev]"

# Or alternatively:
pip install black isort flake8 mypy pytest
```

### 3. Install spaCy Models

We've provided a script to install the required spaCy models:

```bash
# Make the script executable if needed
chmod +x scripts/install_spacy_models.py

# Run the script to install models
./scripts/install_spacy_models.py
```

## VSCode Configuration

### 1. Configure Pylance

Create or update the `.vscode/settings.json` file in your project with the following settings:

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "python.analysis.stubPath": "${workspaceFolder}/stubs",
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.autoImportCompletions": true
}
```

### 2. Configure Python Interpreter

Ensure VSCode is using the correct Python interpreter:

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Python: Select Interpreter"
3. Choose the interpreter for your virtual environment

## Troubleshooting Import Issues

### Issue 1: py_ecc Import Not Resolved

If you still encounter issues with `py_ecc.bn128` imports, try these steps:

1. **Check Stub Files**: Ensure the stub files exist in the correct location:
   - `stubs/py_ecc/__init__.pyi`
   - `stubs/py_ecc/bn128.pyi`

2. **Restart VSCode**: Sometimes a full restart of VSCode is needed for Pylance to recognize new stub files.

3. **Verify pyproject.toml**: Check that `pyproject.toml` includes the stub path configuration:
   ```toml
   [tool.pylance]
   stubPath = "stubs"
   ```

4. **Force a Reload**: Sometimes you can force Pylance to reload by:
   - Pressing `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "Developer: Reload Window"

### Issue 2: spacy Import Not Resolved

For issues with `spacy` imports:

1. **Verify Installation**: Make sure spaCy is actually installed:
   ```bash
   pip show spacy
   ```

2. **Use TYPE_CHECKING**: Use the `TYPE_CHECKING` approach in your code:
   ```python
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       import spacy
   else:
       try:
           import spacy
       except ImportError:
           pass
   ```

3. **Add Inline Comments**: Add `# type: ignore` comments if needed:
   ```python
   import spacy  # type: ignore
   ```

## Advanced Configuration

### Custom Stub Files

We provide custom stub files for some third-party libraries that don't have official type stubs. You can add more as needed:

1. Create appropriate stub files in the `stubs/` directory
2. Make sure they have a `.pyi` extension
3. Follow the library's API in your stub file
4. Restart VSCode to recognize new stubs

### Using pyrightconfig.json (Optional)

For more advanced Pylance configuration, you can create a `pyrightconfig.json` file in your project root:

```json
{
  "include": [
    "llamachain"
  ],
  "exclude": [
    "**/node_modules",
    "**/__pycache__"
  ],
  "typeCheckingMode": "basic",
  "reportMissingImports": true,
  "reportMissingTypeStubs": false,
  "pythonVersion": "3.9",
  "stubPath": "stubs"
}
```

## Additional Resources

- [Pylance Documentation](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Python Type Checking Guide](https://docs.python.org/3/library/typing.html)
- [mypy Documentation](https://mypy.readthedocs.io/) 