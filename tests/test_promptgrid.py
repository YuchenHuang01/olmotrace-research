"""
Tests for prompt grid generation.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from framing_audit.framing_gen import render_prompt_grid, default_prompts


class TestPromptGrid:
    """Test cases for prompt grid generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Synthetic framings
        self.test_framings = {
            'immigration': {
                'positive_neutral': ['immigrants', 'newcomers'],
                'negative_criminalizing': ['illegals'],
                'crisis_alarmist': []
            },
            'taxation': {
                'positive_neutral': ['estate tax'],
                'negative_criminalizing': ['death tax'],
                'crisis_alarmist': []
            }
        }
        
        # Test prompts
        self.test_prompts = {
            'neutral': 'What is the impact of {TERM}?',
            'provocative': 'Why is {TERM} harmful?'
        }
        
        # Create temp directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temp directory."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            
    def test_render_prompt_grid_basic(self):
        """Test basic prompt grid generation without paraphrases."""
        run_id = "test_run"
        
        csv_path = render_prompt_grid(
            framings=self.test_framings,
            prompts=self.test_prompts,
            styles=None,
            run_id=run_id,
            paraphrase=False,
            seed=42
        )
        
        # Check file was created
        assert Path(csv_path).exists()
        
        # Load and validate CSV
        df = pd.read_csv(csv_path)
        
        # Check columns
        expected_columns = [
            'domain', 'term', 'bucket', 'template_key',
            'style_key', 'style_value', 'variant_id', 'prompt_text'
        ]
        assert list(df.columns) == expected_columns
        
        # Check content
        assert len(df) > 0
        assert df['variant_id'].unique().tolist() == [0]  # No paraphrases
        
        # Check all terms are present
        all_terms = []
        for buckets in self.test_framings.values():
            for terms in buckets.values():
                all_terms.extend(terms)
        
        assert set(df['term'].unique()) == set(all_terms)
        
        # Check all templates used
        assert set(df['template_key'].unique()) == set(self.test_prompts.keys())
        
    def test_prompt_text_formatting(self):
        """Test that prompt texts are properly formatted."""
        csv_path = render_prompt_grid(
            framings=self.test_framings,
            prompts=self.test_prompts,
            run_id="format_test",
            paraphrase=False
        )
        
        df = pd.read_csv(csv_path)
        
        # Check no placeholders remain
        for prompt in df['prompt_text']:
            assert '{TERM}' not in prompt
            assert '{' not in prompt or '}' not in prompt
            
        # Check specific formatting
        immigration_neutral = df[
            (df['domain'] == 'immigration') & 
            (df['term'] == 'immigrants') &
            (df['template_key'] == 'neutral')
        ]['prompt_text'].iloc[0]
        
        assert immigration_neutral == 'What is the impact of immigrants?'
        
    def test_prompt_grid_with_styles(self):
        """Test prompt grid with style variations."""
        styles = {
            'audience': ['for policymakers', 'for students'],
            'format': ['in detail', 'briefly']
        }
        
        csv_path = render_prompt_grid(
            framings=self.test_framings,
            prompts=self.test_prompts,
            styles=styles,
            run_id="style_test",
            paraphrase=False
        )
        
        df = pd.read_csv(csv_path)
        
        # Check that some rows have styles
        styled_rows = df[df['style_key'] != '']
        assert len(styled_rows) > 0
        
        # Check style values are from provided options
        for _, row in styled_rows.iterrows():
            if row['style_key']:
                assert row['style_key'] in styles
                assert row['style_value'] in styles[row['style_key']]
                
    def test_guardrails(self):
        """Test that guardrails filter out invalid prompts."""
        # Create prompts with issues
        bad_prompts = {
            'too_short': '{TERM}?',  # Too short
            'too_long': 'This is a very long prompt that asks about {TERM} ' * 10,  # Too long
            'has_placeholder': 'What about {TERM} and {OTHER}?'  # Extra placeholder
        }
        
        csv_path = render_prompt_grid(
            framings={'test': {'positive_neutral': ['test term']}},
            prompts=bad_prompts,
            run_id="guardrail_test",
            paraphrase=False
        )
        
        df = pd.read_csv(csv_path)
        
        # Should have filtered out invalid prompts
        assert len(df) < len(bad_prompts)  # Some should be filtered
        
        # Check remaining prompts meet criteria
        for prompt in df['prompt_text']:
            assert 90 <= len(prompt) <= 220
            assert prompt.count('{') == prompt.count('}') == 0
            
    def test_deduplication(self):
        """Test that duplicate prompts are removed."""
        # Create prompts that will generate duplicates
        duplicate_prompts = {
            'prompt1': 'What is {TERM}?',
            'prompt2': 'What is {TERM}?',  # Exact duplicate
        }
        
        csv_path = render_prompt_grid(
            framings={'test': {'positive_neutral': ['item']}},
            prompts=duplicate_prompts,
            run_id="dedup_test",
            paraphrase=False
        )
        
        df = pd.read_csv(csv_path)
        
        # Check no duplicate prompt_text values
        assert len(df['prompt_text'].unique()) == len(df)
        
    def test_metadata_creation(self):
        """Test that metadata.json is created correctly."""
        run_id = "metadata_test"
        
        render_prompt_grid(
            framings=self.test_framings,
            prompts=self.test_prompts,
            run_id=run_id,
            paraphrase=False,
            seed=123
        )
        
        # Check metadata file exists
        metadata_path = Path(f"runs/{run_id}/metadata.json")
        assert metadata_path.exists()
        
        # Load and validate metadata
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        assert 'timestamp' in metadata
        assert metadata['domains'] == list(self.test_framings.keys())
        assert metadata['seed'] == 123
        assert metadata['paraphrase'] is False
        assert 'decode_params' in metadata
        
    def test_csv_structure(self):
        """Test the structure of the generated CSV."""
        csv_path = render_prompt_grid(
            framings=self.test_framings,
            prompts=self.test_prompts,
            run_id="structure_test",
            paraphrase=False
        )
        
        df = pd.read_csv(csv_path)
        
        # Check data types
        assert df['domain'].dtype == 'object'
        assert df['term'].dtype == 'object'
        assert df['bucket'].dtype == 'object'
        assert df['template_key'].dtype == 'object'
        assert df['variant_id'].dtype == 'int64'
        
        # Check variant_id starts at 0
        assert df['variant_id'].min() == 0
        
        # Check bucket values are valid
        valid_buckets = ['positive_neutral', 'negative_criminalizing', 'crisis_alarmist']
        assert all(bucket in valid_buckets for bucket in df['bucket'].unique())
        
    def test_empty_buckets_handled(self):
        """Test that empty buckets are handled correctly."""
        framings_with_empty = {
            'test_domain': {
                'positive_neutral': ['term1'],
                'negative_criminalizing': [],  # Empty bucket
                'crisis_alarmist': []  # Empty bucket
            }
        }
        
        csv_path = render_prompt_grid(
            framings=framings_with_empty,
            prompts={'simple': 'Tell me about {TERM}.'},
            run_id="empty_bucket_test",
            paraphrase=False
        )
        
        df = pd.read_csv(csv_path)
        
        # Should only have rows for non-empty buckets
        assert len(df) == 1  # One term, one template
        assert df.iloc[0]['term'] == 'term1'
        assert df.iloc[0]['bucket'] == 'positive_neutral'