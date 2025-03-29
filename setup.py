#!/usr/bin/env python
"""
Setup script for LlamaChain.

This script installs the LlamaChain package.
"""

import os
from setuptools import setup, find_packages

# Get version from package
with open(os.path.join("llamachain", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.0.1"

# Get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Parse requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = []
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("# "):
            # Remove version specifiers
            if 'spacy-model' in line:
                continue  # Skip spaCy models as they're installed separately
            requirements.append(line.split('#')[0].strip())

# Define package data - include stubs for better IDE integration
package_data = {
    '': ['py.typed'],  # PEP 561 marker for typed packages
}

# Include stub files for Pylance
data_files = [
    ('stubs/py_ecc', ['stubs/py_ecc/__init__.pyi', 'stubs/py_ecc/bn128.pyi']),
    ('stubs/spacy', ['stubs/spacy/__init__.pyi']),
    ('stubs/spacy/cli', ['stubs/spacy/cli/__init__.pyi']),
    ('stubs/spacy/tokens', ['stubs/spacy/tokens/__init__.pyi']),
]

setup(
    name="llamachain",
    version=version,
    author="LlamaChain Team",
    author_email="info@llamachain.io",
    description="Blockchain intelligence and analytics platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamachain/llamachain",
    packages=find_packages() + ['stubs', 'stubs.py_ecc', 'stubs.spacy', 'stubs.spacy.cli', 'stubs.spacy.tokens'],
    include_package_data=True,
    package_data=package_data,
    data_files=data_files,
    install_requires=[req for req in requirements if not any(x in req for x in ('py-ecc', 'spacy', 'transformers', 'torch'))],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
        ],
        "nlp": [
            "spacy>=3.5.3",
            "transformers>=4.28.1",
            "torch>=2.0.1",
            "nltk>=3.8.1",
            "scikit-learn>=1.2.2",
            "sentence-transformers>=2.2.2",
        ],
        "security": [
            "py-ecc>=6.0.0",
            "slither-analyzer>=0.9.3",
            "mythril>=0.23.15",
        ],
        "all": [  # Meta-dependency for all optional dependencies
            "spacy>=3.5.3",
            "transformers>=4.28.1",
            "torch>=2.0.1",
            "py-ecc>=6.0.0",
            "slither-analyzer>=0.9.3",
            "mythril>=0.23.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamachain=llamachain.cli.main:run_cli",
            "llamachain-nlp=llamachain.nlp.cli:main",
            "llamachain-install-spacy=scripts.install_spacy_models:main",
        ],
    },
) 