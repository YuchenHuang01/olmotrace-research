"""
Tests for framing generation schema and functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from framing_audit.framing_gen import FramingGenerator, generate_framings


class TestFramingGenerationSchema:
    """Test cases for framing generation schema validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_generator = Mock(spec=FramingGenerator)
        
    @patch('framing_audit.framing_gen.FramingGenerator')
    def test_generate_framings_schema(self, mock_generator_class):
        """Test that generate_framings returns correct schema."""
        # Mock the FramingGenerator instance
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        # Mock LLM response for domain generation
        mock_instance.llm_generate_yaml_for_domain.return_value = {
            'positive_neutral': ['undocumented immigrants', 'migrant workers'],
            'negative_criminalizing': ['illegal aliens', 'border jumpers'],
            'crisis_alarmist': ['invasion force']
        }
        
        # Mock other methods
        mock_instance._clean_phrase.side_effect = lambda x: x  # Return unchanged
        mock_instance._is_valid_phrase.return_value = True
        mock_instance._normalize_for_dedup.side_effect = lambda x: x.lower()
        
        # Test generation
        domains = ['immigration']
        result = generate_framings(
            domains=domains,
            model_name='mock-model',
            use_embeddings=False,
            max_per_bucket=6,
            seed=42
        )
        
        # Validate schema
        assert isinstance(result, dict)
        assert 'immigration' in result
        
        domain_framings = result['immigration']
        required_buckets = ['positive_neutral', 'negative_criminalizing', 'crisis_alarmist']
        
        for bucket in required_buckets:
            assert bucket in domain_framings
            assert isinstance(domain_framings[bucket], list)
            
            # All items should be strings
            for phrase in domain_framings[bucket]:
                assert isinstance(phrase, str)
                assert len(phrase.strip()) > 0
        
        # Validate content
        assert len(domain_framings['positive_neutral']) <= 6
        assert len(domain_framings['negative_criminalizing']) <= 6
        assert len(domain_framings['crisis_alarmist']) <= 6
        
    @patch('framing_audit.framing_gen.FramingGenerator')
    def test_generate_framings_with_embeddings(self, mock_generator_class):
        """Test generation with embedding expansion and deduplication."""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        # Mock initial domain generation
        mock_instance.llm_generate_yaml_for_domain.return_value = {
            'positive_neutral': ['term1', 'term2'],
            'negative_criminalizing': ['term3'],
            'crisis_alarmist': []
        }
        
        # Mock variant generation
        mock_instance.llm_list_variants.return_value = [
            'variant1', 'variant2', 'variant3', 'duplicate_term1'
        ]
        
        # Mock embeddings
        mock_embeddings = np.random.rand(7, 384)  # 7 total phrases, 384 dims
        mock_instance.get_sentence_embeddings.return_value = mock_embeddings
        
        # Mock deduplication (simulate removing one duplicate)
        mock_instance._deduplicate_phrases.return_value = [
            'term1', 'term2', 'term3', 'variant1', 'variant2', 'variant3'
        ]
        
        # Mock validation (keep all)
        mock_instance._validate_same_referent.return_value = [
            'term1', 'term2', 'term3', 'variant1', 'variant2', 'variant3'
        ]
        
        # Mock relabeling
        mock_instance._relabel_buckets.return_value = {
            'positive_neutral': ['term1', 'variant1'],
            'negative_criminalizing': ['term3', 'variant2'],
            'crisis_alarmist': ['variant3']
        }
        
        # Mock cleaning methods
        mock_instance._clean_phrase.side_effect = lambda x: x
        mock_instance._is_valid_phrase.return_value = True
        mock_instance._normalize_for_dedup.side_effect = lambda x: x.lower()
        
        # Test with embeddings
        result = generate_framings(
            domains=['test_domain'],
            use_embeddings=True,
            neighbors=4,
            max_per_bucket=3,
            seed=42
        )
        
        # Validate schema
        assert isinstance(result, dict)
        assert 'test_domain' in result
        
        domain_data = result['test_domain']
        for bucket in ['positive_neutral', 'negative_criminalizing', 'crisis_alarmist']:
            assert bucket in domain_data
            assert len(domain_data[bucket]) <= 3  # Respects max_per_bucket
        
    def test_framing_generator_clean_phrase(self):
        """Test phrase cleaning functionality."""
        generator = FramingGenerator()
        
        # Test quote removal
        assert generator._clean_phrase('"test phrase"') == 'test phrase'
        assert generator._clean_phrase("'test phrase'") == 'test phrase'
        assert generator._clean_phrase('`test phrase`') == 'test phrase'
        
        # Test whitespace stripping
        assert generator._clean_phrase('  test phrase  ') == 'test phrase'
        
        # Test empty string
        assert generator._clean_phrase('') == ''
        assert generator._clean_phrase('   ') == ''
        
    def test_framing_generator_is_valid_phrase(self):
        """Test phrase validation."""
        generator = FramingGenerator()
        
        # Valid phrases
        assert generator._is_valid_phrase('immigration policy')
        assert generator._is_valid_phrase('tax')
        assert generator._is_valid_phrase('climate change action plan')  # 4 words
        
        # Invalid phrases
        assert not generator._is_valid_phrase('')  # Empty
        assert not generator._is_valid_phrase('a')  # Too short words
        assert not generator._is_valid_phrase('one two three four five')  # Too many words
        
    def test_framing_generator_normalize_for_dedup(self):
        """Test normalization for deduplication."""
        generator = FramingGenerator()
        
        # Test case normalization
        assert generator._normalize_for_dedup('Test Phrase') == 'test phrase'
        
        # Test punctuation removal
        assert generator._normalize_for_dedup('test, phrase!') == 'test phrase'
        
        # Test whitespace normalization
        assert generator._normalize_for_dedup('test    phrase') == 'test phrase'
        
    @patch('framing_audit.framing_gen.FramingGenerator')
    def test_bucket_size_limits(self, mock_generator_class):
        """Test that bucket size limits are enforced."""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        # Mock generation of more phrases than allowed
        mock_instance.llm_generate_yaml_for_domain.return_value = {
            'positive_neutral': [f'phrase_{i}' for i in range(10)],  # 10 phrases
            'negative_criminalizing': [f'neg_phrase_{i}' for i in range(8)],  # 8 phrases
            'crisis_alarmist': [f'crisis_phrase_{i}' for i in range(5)]  # 5 phrases
        }
        
        mock_instance._clean_phrase.side_effect = lambda x: x
        mock_instance._is_valid_phrase.return_value = True
        mock_instance._normalize_for_dedup.side_effect = lambda x: x.lower()
        
        # Generate with max_per_bucket=3
        result = generate_framings(
            domains=['test'],
            max_per_bucket=3,
            use_embeddings=False,
            seed=42
        )
        
        # Check limits are enforced
        for bucket in result['test'].values():
            assert len(bucket) <= 3
        
    @patch('framing_audit.framing_gen.FramingGenerator')
    def test_cross_bucket_deduplication(self, mock_generator_class):
        """Test that phrases are deduplicated across buckets."""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        # Mock generation with duplicates across buckets
        mock_instance.llm_generate_yaml_for_domain.return_value = {
            'positive_neutral': ['immigration reform', 'border policy'],
            'negative_criminalizing': ['illegal immigration', 'immigration reform'],  # Duplicate
            'crisis_alarmist': ['border crisis']
        }
        
        mock_instance._clean_phrase.side_effect = lambda x: x
        mock_instance._is_valid_phrase.return_value = True
        
        # Mock normalize to detect the duplicate
        def mock_normalize(phrase):
            return phrase.lower().replace(' ', '')
        mock_instance._normalize_for_dedup.side_effect = mock_normalize
        
        result = generate_framings(
            domains=['test'],
            use_embeddings=False,
            seed=42
        )
        
        # Collect all phrases across buckets
        all_phrases = []
        for bucket_phrases in result['test'].values():
            all_phrases.extend(bucket_phrases)
        
        # Check that 'immigration reform' appears only once
        immigration_reform_count = sum(1 for p in all_phrases if 'immigration reform' in p.lower())
        assert immigration_reform_count <= 1