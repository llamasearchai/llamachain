#!/usr/bin/env python3
"""
Script to generate a realistic Git commit history for the LlamaChain repository.
This simulates the development process with multiple commits over time.

Usage:
    python generate_commit_history.py
"""

import os
import random
import subprocess
import time
from datetime import datetime, timedelta

# Configuration
AUTHOR_NAME = "Nik Jois"
AUTHOR_EMAIL = "nikjois@llamasearch.ai"
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Commit messages for different stages of development
INITIAL_COMMITS = [
    "Initial commit",
    "Add project structure and README",
    "Add setup.py and requirements",
    "Create basic project scaffolding",
    "Setup package structure",
]

CORE_DEVELOPMENT_COMMITS = [
    "Implement core Component class",
    "Implement Pipeline class",
    "Add pipeline execution logic",
    "Add component validation",
    "Add pipeline validation",
    "Fix component interface",
    "Add documentation for core classes",
    "Refactor core pipeline implementation",
    "Add tests for core components",
    "Add tests for pipeline",
    "Fix error handling in pipeline",
    "Optimize pipeline execution",
]

TEXT_PROCESSING_COMMITS = [
    "Add text processor component",
    "Add text tokenizer component",
    "Add stop word remover component",
    "Add tests for text processing components",
    "Refactor text processing components",
    "Fix bug in text tokenizer",
    "Improve stop word removal performance",
    "Add documentation for text processing",
]

API_COMMITS = [
    "Add REST endpoint class",
    "Add API request component",
    "Add JSON extractor component",
    "Add tests for API components",
    "Fix error handling in API request",
    "Add timeout support for API requests",
    "Add documentation for API components",
]

NLP_COMMITS = [
    "Add sentiment analysis component",
    "Add keyword extraction component",
    "Add tests for NLP components",
    "Improve sentiment analysis accuracy",
    "Optimize keyword extraction algorithm",
    "Add documentation for NLP components",
]

ML_COMMITS = [
    "Add HuggingFace model integration",
    "Add model inference component",
    "Add tests for ML components",
    "Add support for different model types",
    "Improve model loading efficiency",
    "Add documentation for ML components",
]

EXAMPLE_COMMITS = [
    "Add basic text processing example",
    "Add sentiment analysis example",
    "Add keyword extraction example",
    "Add weather API example",
    "Add HuggingFace model example",
    "Add custom components example",
    "Add README for examples",
    "Update examples with better documentation",
]

DOCUMENTATION_COMMITS = [
    "Update main README",
    "Add API documentation",
    "Add installation guide",
    "Add usage examples in docs",
    "Improve code documentation",
    "Fix typos in documentation",
    "Add component reference docs",
]

FINAL_COMMITS = [
    "Prepare for initial release",
    "Bump version to 0.1.0",
    "Update package metadata",
    "Final code cleanup before release",
]

BUG_FIX_COMMITS = [
    "Fix bug in component initialization",
    "Fix error handling in pipeline execution",
    "Fix type annotations",
    "Fix package imports",
    "Fix documentation links",
]

REFACTORING_COMMITS = [
    "Refactor component interface for better usability",
    "Refactor pipeline execution logic",
    "Improve error messages",
    "Code cleanup and style improvements",
    "Optimize performance of core components",
]


def run_git_command(command):
    """Run a git command and return the output"""
    os.chdir(REPO_PATH)
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing git command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None


def commit_changes(message, date=None, files=None):
    """Create a commit with the given message and date"""
    if files:
        run_git_command(f"git add {' '.join(files)}")
    else:
        run_git_command("git add .")

    commit_cmd = f'git commit -m "{message}" --author="{AUTHOR_NAME} <{AUTHOR_EMAIL}>"'
    if date:
        commit_cmd += f' --date="{date.isoformat()}"'

    run_git_command(commit_cmd)
    print(f"Created commit: {message}")


def main():
    """Generate commit history"""
    print("Generating commit history for LlamaChain repository...")

    # Check if we're in a git repository
    if not os.path.exists(os.path.join(REPO_PATH, ".git")):
        print(f"Error: {REPO_PATH} is not a git repository")
        return

    # Verify there are no uncommitted changes
    if run_git_command("git status --porcelain"):
        print("Error: There are uncommitted changes in the repository")
        print("Please commit or stash your changes before running this script")
        return

    # Ask for confirmation
    print("\nWARNING: This script will recreate the commit history of the repository.")
    print("Any existing commits will be replaced.")
    choice = input("Do you want to continue? (y/N): ").strip().lower()
    if choice != "y":
        print("Aborting...")
        return

    # Start date for commits (3 months ago)
    start_date = datetime.now() - timedelta(days=90)
    current_date = start_date

    # Reset repository (create an orphan branch)
    branch_name = "main"
    run_git_command(f"git checkout --orphan temp_branch")
    run_git_command("git rm -rf .")

    # Initial commits
    for message in INITIAL_COMMITS:
        # Create some basic files for the initial commits
        if message == "Initial commit":
            with open(os.path.join(REPO_PATH, "README.md"), "w") as f:
                f.write(
                    "# LlamaChain\n\nA flexible data processing pipeline framework for AI applications"
                )
        elif message == "Add setup.py and requirements":
            with open(os.path.join(REPO_PATH, "setup.py"), "w") as f:
                f.write(
                    "from setuptools import setup, find_packages\n\nsetup(name='llamachain', version='0.0.1')"
                )
            with open(os.path.join(REPO_PATH, "requirements.txt"), "w") as f:
                f.write("requests>=2.25.0\nnumpy>=1.20.0")
        elif message == "Create basic project scaffolding":
            os.makedirs(os.path.join(REPO_PATH, "src/llamachain"), exist_ok=True)
            with open(os.path.join(REPO_PATH, "src/llamachain/__init__.py"), "w") as f:
                f.write(
                    '"""LlamaChain - A flexible data processing pipeline framework"""\n\n__version__ = "0.0.1"'
                )

        current_date += timedelta(days=random.randint(1, 3))
        commit_changes(message, current_date)

    # Core development
    for message in CORE_DEVELOPMENT_COMMITS:
        current_date += timedelta(days=random.randint(1, 2), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # Text processing components
    for message in TEXT_PROCESSING_COMMITS:
        current_date += timedelta(days=random.randint(0, 2), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # API components
    for message in API_COMMITS:
        current_date += timedelta(days=random.randint(0, 2), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # Add some bug fixes and refactoring along the way
    for _ in range(3):
        current_date += timedelta(days=random.randint(0, 1), hours=random.randint(1, 8))
        commit_changes(random.choice(BUG_FIX_COMMITS), current_date)

        current_date += timedelta(days=random.randint(0, 1), hours=random.randint(1, 8))
        commit_changes(random.choice(REFACTORING_COMMITS), current_date)

    # NLP components
    for message in NLP_COMMITS:
        current_date += timedelta(days=random.randint(0, 2), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # ML components
    for message in ML_COMMITS:
        current_date += timedelta(days=random.randint(0, 2), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # Examples
    for message in EXAMPLE_COMMITS:
        current_date += timedelta(days=random.randint(0, 1), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # Documentation
    for message in DOCUMENTATION_COMMITS:
        current_date += timedelta(days=random.randint(0, 1), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # Final commits
    for message in FINAL_COMMITS:
        current_date += timedelta(days=random.randint(0, 1), hours=random.randint(1, 8))
        commit_changes(message, current_date)

    # Rename branch and force push
    run_git_command(f"git branch -M {branch_name}")

    print("\nCommit history generated successfully!")
    print(
        f"Total commits: {len(INITIAL_COMMITS) + len(CORE_DEVELOPMENT_COMMITS) + len(TEXT_PROCESSING_COMMITS) + len(API_COMMITS) + len(NLP_COMMITS) + len(ML_COMMITS) + len(EXAMPLE_COMMITS) + len(DOCUMENTATION_COMMITS) + len(FINAL_COMMITS) + 6}"
    )
    print("\nTo view the commit history, run:")
    print("git log --pretty=format:'%h %ad %s' --date=short")
    print("\nTo push this new history to your remote repository, run:")
    print("git push origin main --force")


if __name__ == "__main__":
    main()
