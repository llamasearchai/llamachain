#!/usr/bin/env python3
"""
spaCy Model Installer

This script handles the installation of spaCy models required by the LlamaChain NLP module.
It checks if models are already installed and installs them if needed.
"""

import os
import sys
import subprocess
import logging
from typing import List, Dict, Optional, Set, TYPE_CHECKING, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("spacy_model_installer")

# Define SPACY_AVAILABLE at module level so Pylance recognizes it
SPACY_AVAILABLE = False

# Import spaCy only during runtime, not during type checking
# This helps Pylance understand the code even if spaCy isn't installed yet
if TYPE_CHECKING:
    import spacy  # type: ignore
    from spacy.cli import download  # type: ignore
else:
    try:
        import spacy  # type: ignore
        from spacy.cli import download  # type: ignore
        SPACY_AVAILABLE = True
    except ImportError:
        logger.warning("spaCy not installed. Will attempt to install it first.")


# Default models to install
DEFAULT_MODELS = [
    "en_core_web_sm",  # Small English model
    "en_core_web_md",  # Medium English model (better for NLP tasks)
]


def check_installed_models() -> Set[str]:
    """
    Check which spaCy models are already installed.
    
    Returns:
        Set of installed model names
    """
    if not SPACY_AVAILABLE:
        return set()
    
    installed_models = set()
    try:
        # Get the data path where models are stored
        data_path = spacy.util.get_data_path()  # type: ignore
        
        # Models are directories in the data path
        for item in os.listdir(data_path):
            model_path = os.path.join(data_path, item)
            if os.path.isdir(model_path):
                installed_models.add(item)
        
        logger.info(f"Found {len(installed_models)} installed models: {', '.join(installed_models)}")
        return installed_models
    
    except Exception as e:
        logger.error(f"Error checking installed models: {e}")
        return set()


def install_models(models: Optional[List[str]] = None) -> bool:
    """
    Install spaCy models.
    
    Args:
        models: List of model names to install. If None, installs default models.
        
    Returns:
        True if successful, False otherwise
    """
    models = models or DEFAULT_MODELS
    
    # First ensure spaCy is installed
    if not SPACY_AVAILABLE:
        logger.info("Installing spaCy...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy>=3.5.0,<4.0.0"])
            
            # Now try to import spaCy again
            try:
                import spacy  # type: ignore
                from spacy.cli import download  # type: ignore
                global SPACY_AVAILABLE
                SPACY_AVAILABLE = True
                logger.info("spaCy successfully installed")
            except ImportError as e:
                logger.error(f"Failed to import spaCy after installation: {e}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install spaCy: {e}")
            return False
    
    # Check which models are already installed
    installed_models = check_installed_models()
    models_to_install = [model for model in models if model not in installed_models]
    
    if not models_to_install:
        logger.info("All required models are already installed")
        return True
    
    # Install each model
    success = True
    for model in models_to_install:
        logger.info(f"Installing model: {model}")
        try:
            download(model)  # type: ignore
            logger.info(f"Successfully installed {model}")
        except Exception as e:
            logger.error(f"Failed to install {model}: {e}")
            success = False
    
    return success


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install spaCy models for LlamaChain")
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specify models to install (default: en_core_web_sm en_core_web_md)"
    )
    
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS
    
    logger.info(f"Preparing to install the following spaCy models: {', '.join(models)}")
    
    success = install_models(models)
    
    if success:
        logger.info("All models installed successfully")
        sys.exit(0)
    else:
        logger.error("Failed to install some models")
        sys.exit(1)


if __name__ == "__main__":
    main() 