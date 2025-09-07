"""
Tests for prompt template generation.
"""

import pytest
from framing_audit.framing_gen import generate_prompts


class TestPromptsGeneration:
    """Test cases for prompt template generation."""
    
    def test_generate_prompts_structure(self):
        """Test that generate_prompts returns correct structure."""
        prompts = generate_prompts()
        
        # Check that it returns a dictionary
        assert isinstance(prompts, dict)
        
        # Check required prompt types are present
        required_prompts = [
            'neutral',
            'provocative', 
            'factual',
            'human_rights',
            'security',
            'empathy'
        ]
        
        for prompt_type in required_prompts:
            assert prompt_type in prompts, f"Missing required prompt type: {prompt_type}"
            
    def test_prompts_contain_placeholder(self):
        """Test that all prompts contain the {TERM} placeholder."""
        prompts = generate_prompts()
        
        for prompt_name, prompt_template in prompts.items():
            assert isinstance(prompt_template, str), f"Prompt {prompt_name} is not a string"
            assert '{TERM}' in prompt_template, f"Prompt {prompt_name} missing {{TERM}} placeholder"
            assert len(prompt_template.strip()) > 0, f"Prompt {prompt_name} is empty"
            
    def test_prompts_content_validity(self):
        """Test that prompts contain reasonable content."""
        prompts = generate_prompts()
        
        # Test specific prompt characteristics
        assert 'economy' in prompts['neutral'].lower()
        assert 'threat' in prompts['provocative'].lower()
        assert 'policy' in prompts['factual'].lower() or 'debate' in prompts['factual'].lower()
        assert 'rights' in prompts['human_rights'].lower() or 'ethical' in prompts['human_rights'].lower()
        assert 'security' in prompts['security'].lower()
        assert 'compassion' in prompts['empathy'].lower() or 'challenge' in prompts['empathy'].lower()
        
    def test_prompts_are_questions_or_statements(self):
        """Test that prompts are properly formatted as questions or statements."""
        prompts = generate_prompts()
        
        for prompt_name, prompt_template in prompts.items():
            # Should end with proper punctuation
            assert prompt_template.strip().endswith(('.', '?', '!')), \
                f"Prompt {prompt_name} doesn't end with proper punctuation"
                
            # Should start with capital letter or common question words
            first_word = prompt_template.strip().split()[0]
            assert first_word[0].isupper() or first_word.lower() in ['what', 'why', 'how', 'when', 'where'], \
                f"Prompt {prompt_name} doesn't start properly: {first_word}"
                
    def test_prompts_uniqueness(self):
        """Test that all prompts are unique."""
        prompts = generate_prompts()
        
        prompt_texts = list(prompts.values())
        unique_prompts = set(prompt_texts)
        
        assert len(prompt_texts) == len(unique_prompts), "Some prompts are duplicated"
        
    def test_prompt_length_reasonable(self):
        """Test that prompts are reasonable length."""
        prompts = generate_prompts()
        
        for prompt_name, prompt_template in prompts.items():
            word_count = len(prompt_template.split())
            assert 5 <= word_count <= 30, \
                f"Prompt {prompt_name} has unreasonable length: {word_count} words"
                
    def test_placeholder_substitution(self):
        """Test that placeholder can be properly substituted."""
        prompts = generate_prompts()
        
        test_term = "immigration policy"
        
        for prompt_name, prompt_template in prompts.items():
            substituted = prompt_template.replace('{TERM}', test_term)
            
            # Check substitution worked
            assert '{TERM}' not in substituted, f"Placeholder not fully replaced in {prompt_name}"
            assert test_term in substituted, f"Test term not found in substituted {prompt_name}"
            
            # Check result is grammatically reasonable
            # Should not have double spaces or weird punctuation
            assert '  ' not in substituted, f"Double spaces in substituted {prompt_name}"
            assert substituted == substituted.strip(), f"Extra whitespace in substituted {prompt_name}"
            
    def test_prompts_yaml_serializable(self):
        """Test that prompts can be serialized to YAML."""
        import yaml
        
        prompts = generate_prompts()
        
        try:
            yaml_str = yaml.dump(prompts, default_flow_style=False)
            # Try to load it back
            reloaded = yaml.safe_load(yaml_str)
            
            assert reloaded == prompts, "YAML serialization/deserialization changed data"
            
        except Exception as e:
            pytest.fail(f"Failed to serialize prompts to YAML: {e}")
            
    def test_specific_prompt_content(self):
        """Test specific expected content in prompts."""
        prompts = generate_prompts()
        
        # Test neutral prompt is actually neutral
        neutral = prompts['neutral'].lower()
        assert 'impact' in neutral and 'economy' in neutral
        
        # Test provocative prompt has provocative language
        provocative = prompts['provocative'].lower()
        assert 'why' in provocative and 'threat' in provocative
        
        # Test factual prompt asks for factual information
        factual = prompts['factual'].lower()
        assert any(word in factual for word in ['summarize', 'debate', 'policy'])
        
        # Test human rights prompt mentions rights/ethics
        human_rights = prompts['human_rights'].lower()
        assert any(word in human_rights for word in ['rights', 'ethical', 'considerations'])
        
        # Test security prompt mentions security
        security = prompts['security'].lower()
        assert 'security' in security
        
        # Test empathy prompt has compassionate framing
        empathy = prompts['empathy'].lower()
        assert any(word in empathy for word in ['compassionate', 'challenges', 'people', 'affected'])