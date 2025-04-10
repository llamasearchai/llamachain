#!/usr/bin/env python
"""
LlamaChain setup module
"""

import os
from setuptools import setup, find_packages

# Get package version from src/llamachain/__init__.py
version = {}
with open(os.path.join("src", "llamachain", "__init__.py")) as f:
    exec(f.read(), version)

# Read README for long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
REQUIRED = [
    "requests>=2.25.0",
    "numpy>=1.20.0",
]

# Optional dependencies
EXTRAS = {
    "ml": [
        "torch>=1.9.0",
        "transformers>=4.5.0",
    ],
    "nlp": [
        "spacy>=3.0.0",
    ],
    "viz": [
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
    ],
    "dev": [
        "pytest>=6.0.0",
        "black>=21.5b2",
        "flake8>=3.9.0",
        "mypy>=0.812",
        "isort>=5.8.0",
        "pytest-cov>=2.12.0",
    ],
    "docs": [
        "mkdocs>=1.2.0",
        "mkdocs-material>=7.1.0",
        "mkdocstrings>=0.15.0",
    ],
}

# Full dependencies
EXTRAS["all"] = [pkg for group in EXTRAS.values() for pkg in group]

setup(
    name="llamachain",
    version=version.get("__version__", "0.1.0"),
    description="A flexible data processing pipeline framework for AI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    url="https://github.com/llamasearchai/llamachain",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8.0",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pipeline, data processing, machine learning, AI, framework",
    project_urls={
        "Documentation": "https://llamasearch.ai/docs/llamachain",
        "Source": "https://github.com/llamasearchai/llamachain",
        "Issue Tracker": "https://github.com/llamasearchai/llamachain/issues",
    },
)
