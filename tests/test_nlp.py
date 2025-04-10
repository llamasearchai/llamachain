"""
Tests for the NLP module.
"""

import pytest
from unittest.mock import patch, MagicMock

from llamachain.nlp.processor import NLPProcessor
from llamachain.nlp.intent import Intent, IntentClassifier
from llamachain.nlp.entity import Entity, EntityExtractor


@pytest.fixture
def nlp_processor():
    """Create an NLP processor for testing."""
    with patch('llamachain.nlp.processor.IntentClassifier') as mock_intent_classifier, \
         patch('llamachain.nlp.processor.EntityExtractor') as mock_entity_extractor:
        
        # Mock the classify method
        mock_intent = MagicMock()
        mock_intent.to_dict.return_value = {
            "type": "get_transaction",
            "description": "Retrieve transaction information"
        }
        mock_intent_classifier.return_value.classify.return_value = mock_intent
        
        # Mock the extract method
        mock_entity = MagicMock()
        mock_entity.to_dict.return_value = {
            "type": "tx_hash",
            "value": "0x123456789abcdef",
            "start": 0,
            "end": 18,
            "confidence": 0.9
        }
        mock_entity_extractor.return_value.extract.return_value = [mock_entity]
        
        processor = NLPProcessor()
        
        # Replace the _analyze_with_spacy method
        processor._analyze_with_spacy = MagicMock(return_value={
            "entities": [],
            "noun_chunks": [],
            "tokens": []
        })
        
        yield processor


@pytest.mark.asyncio
async def test_process_query(nlp_processor):
    """Test processing a query."""
    # Process a query
    result = await nlp_processor.process_query("What is the transaction 0x123456789abcdef?")
    
    # Check the result
    assert result["original_query"] == "What is the transaction 0x123456789abcdef?"
    assert "cleaned_query" in result
    assert result["intent"]["type"] == "get_transaction"
    assert len(result["entities"]) == 1
    assert result["entities"][0]["type"] == "tx_hash"
    assert result["entities"][0]["value"] == "0x123456789abcdef"


@pytest.mark.asyncio
async def test_generate_response(nlp_processor):
    """Test generating a response."""
    # Create a processed query
    processed_query = {
        "original_query": "What is the transaction 0x123456789abcdef?",
        "cleaned_query": "what is the transaction 0x123456789abcdef",
        "intent": {
            "type": "get_transaction",
            "description": "Retrieve transaction information"
        },
        "entities": [
            {
                "type": "tx_hash",
                "value": "0x123456789abcdef",
                "start": 0,
                "end": 18,
                "confidence": 0.9
            }
        ],
        "spacy_analysis": None
    }
    
    # Generate a response
    response = await nlp_processor.generate_response(processed_query)
    
    # Check the response
    assert response == "I'll retrieve transaction information for you."


@pytest.mark.asyncio
async def test_translate_to_query(nlp_processor):
    """Test translating to a structured query."""
    # Create a processed query
    processed_query = {
        "original_query": "What is the transaction 0x123456789abcdef?",
        "cleaned_query": "what is the transaction 0x123456789abcdef",
        "intent": {
            "type": "get_transaction",
            "description": "Retrieve transaction information"
        },
        "entities": [
            {
                "type": "tx_hash",
                "value": "0x123456789abcdef",
                "start": 0,
                "end": 18,
                "confidence": 0.9
            }
        ],
        "spacy_analysis": None
    }
    
    # Translate to a structured query
    structured_query = await nlp_processor.translate_to_query(processed_query)
    
    # Check the structured query
    assert structured_query["type"] == "get_transaction"
    assert structured_query["params"]["tx_hash"] == "0x123456789abcdef"
    assert structured_query["params"]["blockchain"] == "ethereum"


@pytest.mark.asyncio
async def test_empty_query(nlp_processor):
    """Test processing an empty query."""
    # Process an empty query
    with pytest.raises(Exception):
        await nlp_processor.process_query("")


class TestIntentClassifier:
    """Tests for the IntentClassifier class."""
    
    @pytest.fixture
    def intent_classifier(self):
        """Create an intent classifier for testing."""
        return IntentClassifier()
    
    @pytest.mark.asyncio
    async def test_classify(self, intent_classifier):
        """Test classifying a query."""
        # Classify a query
        intent = await intent_classifier.classify("What is the transaction 0x123456789abcdef?")
        
        # Check the intent
        assert intent == Intent.GET_TRANSACTION
    
    @pytest.mark.asyncio
    async def test_classify_unknown(self, intent_classifier):
        """Test classifying a query with an unknown intent."""
        # Classify a query
        intent = await intent_classifier.classify("Hello, how are you?")
        
        # Check the intent
        assert intent == Intent.UNKNOWN
    
    def test_get_confidence_scores(self, intent_classifier):
        """Test getting confidence scores."""
        # Get confidence scores
        scores = intent_classifier.get_confidence_scores("What is the transaction 0x123456789abcdef?")
        
        # Check the scores
        assert Intent.GET_TRANSACTION in scores
        assert scores[Intent.GET_TRANSACTION] > 0


class TestEntityExtractor:
    """Tests for the EntityExtractor class."""
    
    @pytest.fixture
    def entity_extractor(self):
        """Create an entity extractor for testing."""
        return EntityExtractor()
    
    @pytest.mark.asyncio
    async def test_extract(self, entity_extractor):
        """Test extracting entities."""
        # Extract entities
        entities = await entity_extractor.extract(
            "What is the transaction 0x123456789abcdef?",
            Intent.GET_TRANSACTION
        )
        
        # Check the entities
        assert len(entities) > 0
        assert any(entity.type == "tx_hash" for entity in entities)
    
    @pytest.mark.asyncio
    async def test_extract_address(self, entity_extractor):
        """Test extracting an address entity."""
        # Extract entities
        entities = await entity_extractor.extract(
            "What is the balance of 0x742d35Cc6634C0532925a3b844Bc454e4438f44e?",
            Intent.GET_BALANCE
        )
        
        # Check the entities
        assert len(entities) > 0
        assert any(entity.type == "address" for entity in entities)
    
    @pytest.mark.asyncio
    async def test_extract_token(self, entity_extractor):
        """Test extracting a token entity."""
        # Extract entities
        entities = await entity_extractor.extract(
            "What is the price of ETH?",
            Intent.GET_PRICE
        )
        
        # Check the entities
        assert len(entities) > 0
        assert any(entity.type == "token" for entity in entities)
    
    def test_remove_overlapping_entities(self, entity_extractor):
        """Test removing overlapping entities."""
        # Create overlapping entities
        entities = [
            Entity(type="token", value="ETH", start=0, end=3, confidence=0.9),
            Entity(type="token", value="ETH", start=0, end=3, confidence=0.8),
        ]
        
        # Remove overlapping entities
        result = entity_extractor._remove_overlapping_entities(entities)
        
        # Check the result
        assert len(result) == 1
        assert result[0].confidence == 0.9 