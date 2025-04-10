"""
HuggingFace model integration components for LlamaChain
"""

import logging
from typing import Any, Dict, List, Optional, Union

from llamachain.core import Component


class HuggingFaceModel:
    """Wrapper for HuggingFace models"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "text-classification",
        use_gpu: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize HuggingFace model
        
        Args:
            model_name: Name of the HuggingFace model
            model_type: Type of model ("text-classification", "token-classification", etc.)
            use_gpu: Whether to use GPU if available
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        # Informational message about dependencies
        logging.info("Note: Using HuggingFaceModel requires installing transformers: pip install transformers")
    
    def _lazy_init(self):
        """Lazy initialization of the model on first use"""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
            import torch
        except ImportError:
            raise ImportError(
                "To use HuggingFaceModel, you need to install the transformers package. "
                "Run: pip install transformers torch"
            )
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        
        # Load model and tokenizer based on model type
        if self.model_type == "text-classification":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        
        elif self.model_type == "token-classification":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Move model to device
        self.model.to(self.device)
        self._initialized = True
    
    def predict(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Run inference with the model
        
        Args:
            text: Input text or list of texts
            
        Returns:
            List of prediction results
        """
        self._lazy_init()
        
        try:
            import torch
        except ImportError:
            raise ImportError("To use HuggingFaceModel, you need to install PyTorch: pip install torch")
        
        # Prepare inputs
        inputs = self.tokenizer(
            text if isinstance(text, list) else [text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs based on model type
        if self.model_type == "text-classification":
            # Get raw scores
            logits = outputs.logits.cpu().numpy()
            
            # Convert to probabilities
            import numpy as np
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            
            # Get predicted labels
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            # Get label mapping if available
            label_map = getattr(self.model.config, "id2label", {})
            
            # Create results list
            results = []
            for i, pred in enumerate(predictions):
                label = label_map.get(int(pred), str(pred))
                score = float(probs[i, pred])
                
                results.append({
                    "label": label,
                    "score": score,
                    "all_scores": {label_map.get(j, str(j)): float(probs[i, j]) for j in range(len(probs[i]))}
                })
            
            return results
        
        elif self.model_type == "token-classification":
            # Process token classification output
            # This would be more complex in a real implementation
            logging.warning("Token classification output processing is simplified in this implementation")
            
            # Get predicted labels
            predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()
            
            # Get label mapping if available
            label_map = getattr(self.model.config, "id2label", {})
            
            # Process tokens and predictions
            results = []
            batch_size = predictions.shape[0]
            
            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[i])
                token_predictions = []
                
                for j, token in enumerate(tokens):
                    if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                        continue
                        
                    pred_id = predictions[i, j]
                    label = label_map.get(int(pred_id), str(pred_id))
                    
                    token_predictions.append({
                        "token": token,
                        "label": label
                    })
                
                results.append({"tokens": token_predictions})
            
            return results


class ModelInference(Component):
    """Component for running inference with ML models"""
    
    def __init__(
        self,
        model: Any,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ModelInference
        
        Args:
            model: Model to use for inference
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.model = model
    
    def process(self, input_data: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run inference with the model
        
        Args:
            input_data: Input text or list of texts
            
        Returns:
            Model prediction results
        """
        # Run prediction
        results = self.model.predict(input_data)
        
        # For single input, return first result
        if isinstance(input_data, str):
            return results[0]
        
        # For list input, return all results
        return results 