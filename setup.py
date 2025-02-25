#!/usr/bin/env python
"""
Setup script for LlamaChain.

This script installs the LlamaChain package.
"""

import os
from setuptools import setup, find_packages

# Get version from package
try:
    version_file = os.path.join("src", "llamachain", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break
            else:
                version = "0.1.0"
    else:
        version = "0.1.0"
except Exception:
    version = "0.1.0"

# Get long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except Exception:
    long_description = "LlamaChain - Blockchain intelligence and analytics platform"

# Parse requirements.txt
requirements = [
    "requests>=2.28.2",
    "aiohttp>=3.8.4",
    "pydantic>=1.10.7",
    "web3>=6.0.0",
    "eth-account>=0.8.0",
    "eth-abi>=4.0.0",
    "eth-typing>=3.3.0",
]

try:
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("# "):
                    # Remove version specifiers
                    if 'spacy-model' in line:
                        continue  # Skip spaCy models as they're installed separately
                    requirements.append(line.split('#')[0].strip())
except Exception:
    # If requirements.txt can't be read, use the default requirements defined above
    pass

# Define package data - include stubs for better IDE integration
package_data = {
    '': ['py.typed'],  # PEP 561 marker for typed packages
}

setup(
    name="llamachain-llamasearch",
    version=version,
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    description="Blockchain intelligence and analytics platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data=package_data,
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
# Updated in commit 5 - 2025-04-04 17:41:14

# Updated in commit 13 - 2025-04-04 17:41:14

# Updated in commit 21 - 2025-04-04 17:41:15
