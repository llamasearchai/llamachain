#!/usr/bin/env python3
"""
Setup script for LlamaChain
"""

import os

from setuptools import find_packages, setup

# Get package version
about = {}
with open(os.path.join("src", "llamachain", "__init__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

# Get the long description from the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define dependencies
REQUIRES = ["numpy>=1.19.0", "pandas>=1.0.0", "requests>=2.25.0"]

EXTRAS_REQUIRE = {
    "ml": [
        "torch>=1.7.0",
        "transformers>=4.0.0",
    ],
    "viz": [
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
    ],
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "flake8>=3.8.0",
        "mypy>=0.812",
        "black>=21.5b2",
    ],
    "docs": [
        "mkdocs>=1.1.0",
        "mkdocs-material>=7.0.0",
        "mkdocstrings>=0.15.0",
    ],
}

# Add an "all" option that installs all extras
EXTRAS_REQUIRE["all"] = [
    package for packages in EXTRAS_REQUIRE.values() for package in packages
]

setup(
    name="llamachain",
    version=about["__version__"],
    description="A flexible data processing pipeline framework for AI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__email__"],
    url="https://github.com/llamasearchai/llamachain",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning, natural language processing, data pipeline, ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)
