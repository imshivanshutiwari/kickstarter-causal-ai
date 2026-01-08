
import sys
from unittest.mock import MagicMock
import pytest
from pathlib import Path

# -----------------------------------------------------------------------------
# AGGRESSIVE MOCKING: Must be done BEFORE importing src.nlp_features
# This prevents Windows DLL 0xc0000139 crashes by skipping 'torch'/'transformers' load
# -----------------------------------------------------------------------------
mock_torch = MagicMock()
mock_st = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["sentence_transformers"] = mock_st
sys.modules["sentence_transformers.SentenceTransformer"] = mock_st

# Now safe to import src
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai_consultant import AIConsultant
# We import nlp_features after mocking. usage of torch inside will use the mock
import nlp_features 
from nlp_features import ContentEmbedder

class TestAIConsultant:
    """Tests for the AI Consultant module."""
    
    @pytest.fixture
    def consultant(self):
        return AIConsultant(api_key=None)
    
    @pytest.fixture
    def sample_inputs(self):
        campaign = {'category': 'Technology', 'funding_goal': 10000, 'campaign_duration': 30, 'avg_reward_price': 50, 'goal_ambition': 1.0}
        prediction = {'success_probability': 0.8, 'funding_ratio': 1.2, 'confidence_low': 8000, 'confidence_high': 15000}
        counterfactual = {'price_effect': 'Higher prices increase funding', 'optimal_price': 60}
        return campaign, prediction, counterfactual

    def test_advice_structure(self, consultant, sample_inputs):
        """Ensure advice dictionary has all required keys."""
        advice = consultant.generate_advice(*sample_inputs)
        required_keys = ['headline', 'summary', 'strategies', 'risk_score', 'tone']
        for key in required_keys:
            assert key in advice

class TestNLPFeatures:
    """Tests for the Deep NLP module (Mocked)."""
    
    def test_embedder_flow(self):
        """Test ContentEmbedder flow without loading real models."""
        # Setup the mock instance returned by the class constructor
        mock_model_instance = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model_instance
        
        # Mock what .encode() returns (numpy array simulation)
        import numpy as np
        mock_model_instance.encode.return_value = np.zeros((2, 384))
        
        embedder = ContentEmbedder()
        texts = ["A", "B"]
        embeddings = embedder.generate_embeddings(texts)
        
        assert embeddings.shape == (2, 384)
        mock_st.SentenceTransformer.assert_called_once()
