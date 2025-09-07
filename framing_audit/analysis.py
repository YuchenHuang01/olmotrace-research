"""
Analysis module for bias metrics and domain labeling.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter, defaultdict
from transformers import pipeline
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)


class BiasAnalyzer:
    """Analyzes bias and source characteristics in generated text and training data."""
    
    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment",
        toxicity_threshold: float = 0.7,
    ):
        """
        Initialize bias analyzer.
        
        Args:
            sentiment_model: HuggingFace model for sentiment analysis
            toxicity_threshold: Threshold for toxicity classification
        """
        self.toxicity_threshold = toxicity_threshold
        
        # Initialize sentiment pipeline
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                device=-1  # CPU
            )
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}")
            self.sentiment_analyzer = None
            
        # Domain patterns for source bucketing
        self.domain_patterns = {
            'wikipedia': [
                r'wikipedia\.org',
                r'wiki[mp]edia',
                r'encyclop[ae]dia'
            ],
            'mainstream_news': [
                r'\b(nytimes|washingtonpost|wsj|reuters|apnews|bbc|cnn|foxnews|msnbc|npr)\b',
                r'\b(times|post|journal|news|press|tribune|herald)\b',
                r'associated press',
                r'news agency'
            ],
            'gov_ngo': [
                r'\.gov\b',
                r'\b(government|federal|state|department|agency|bureau)\b',
                r'\b(ngo|non-?profit|foundation|institute|organization)\b',
                r'\b(united nations|who|unesco|unicef)\b'
            ],
            'blog': [
                r'\b(blog|blogger|wordpress|medium\.com|substack)\b',
                r'personal (blog|website|page)',
                r'\b(opinion|editorial|commentary)\b'
            ],
            'forum': [
                r'\b(reddit|twitter|facebook|forum|discussion|comment)\b',
                r'posted by',
                r'replied to',
                r'\b(thread|post|reply)\b'
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {
            domain: [re.compile(pattern, re.IGNORECASE) 
                    for pattern in patterns]
            for domain, patterns in self.domain_patterns.items()
        }
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores
        """
        if self.sentiment_analyzer and text.strip():
            try:
                results = self.sentiment_analyzer(text[:512])  # Limit length
                
                # Convert to standardized format
                sentiment_map = {
                    'POSITIVE': 1.0,
                    'NEGATIVE': -1.0,
                    'NEUTRAL': 0.0
                }
                
                label = results[0]['label'].upper()
                score = results[0]['score']
                
                return {
                    'sentiment': sentiment_map.get(label, 0.0),
                    'confidence': score
                }
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                
        # Fallback to TextBlob
        try:
            blob = TextBlob(text)
            return {
                'sentiment': blob.sentiment.polarity,
                'confidence': abs(blob.sentiment.polarity)
            }
        except:
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    def analyze_toxicity(self, text: str) -> Dict[str, float]:
        """
        Analyze toxicity of text using heuristics.
        
        Args:
            text: Input text
            
        Returns:
            Dict with toxicity scores
        """
        # Simple heuristic-based toxicity detection
        toxic_patterns = [
            r'\b(hate|kill|destroy|attack|threat|dangerous|invasion|criminal)\b',
            r'\b(stupid|idiot|moron|dumb|fool)\b',
            r'\b(disgusting|vile|evil|corrupt)\b',
            r'[!]{3,}',  # Multiple exclamation marks
            r'[A-Z]{5,}',  # All caps shouting
        ]
        
        text_lower = text.lower()
        toxicity_score = 0.0
        
        for pattern in toxic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                toxicity_score += 0.2
                
        # Cap at 1.0
        toxicity_score = min(1.0, toxicity_score)
        
        return {
            'toxicity': toxicity_score,
            'is_toxic': toxicity_score >= self.toxicity_threshold
        }
    
    def analyze_subjectivity(self, text: str) -> Dict[str, float]:
        """
        Analyze subjectivity of text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'subjectivity': blob.sentiment.subjectivity,
                'is_subjective': blob.sentiment.subjectivity > 0.5
            }
        except:
            return {'subjectivity': 0.5, 'is_subjective': False}
    
    def classify_domain(self, snippet_text: str) -> str:
        """
        Classify the domain/source type of a snippet.
        
        Args:
            snippet_text: Text snippet
            
        Returns:
            Domain label
        """
        text_lower = snippet_text.lower()
        
        # Check each domain's patterns
        for domain, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return domain
                    
        return 'other'
    
    def analyze_output_metrics(self, text: str) -> Dict[str, Any]:
        """
        Analyze multiple metrics for generated output text.
        
        Args:
            text: Generated text
            
        Returns:
            Dict of metrics
        """
        sentiment = self.analyze_sentiment(text)
        toxicity = self.analyze_toxicity(text)
        subjectivity = self.analyze_subjectivity(text)
        
        return {
            'sentiment': sentiment['sentiment'],
            'sentiment_confidence': sentiment['confidence'],
            'toxicity': toxicity['toxicity'],
            'is_toxic': toxicity['is_toxic'],
            'subjectivity': subjectivity['subjectivity'],
            'is_subjective': subjectivity['is_subjective'],
            'length': len(text.split()),
            'coverage': 1.0  # Placeholder for span coverage
        }
    
    def analyze_source_distribution(
        self,
        snippets: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Analyze distribution of source domains in snippets.
        
        Args:
            snippets: List of snippet dictionaries
            
        Returns:
            Count of snippets per domain
        """
        domain_counts = Counter()
        
        for snippet in snippets:
            snippet_text = snippet.get('snippet_text', '')
            domain = self.classify_domain(snippet_text)
            domain_counts[domain] += 1
            
        return dict(domain_counts)
    
    def compute_jaccard_overlap(
        self,
        doc_set1: Set[Tuple[str, str]],
        doc_set2: Set[Tuple[str, str]]
    ) -> float:
        """
        Compute Jaccard similarity between two sets of documents.
        
        Args:
            doc_set1: Set of (index_name, doc_ptr) tuples
            doc_set2: Set of (index_name, doc_ptr) tuples
            
        Returns:
            Jaccard similarity score
        """
        if not doc_set1 and not doc_set2:
            return 1.0
        if not doc_set1 or not doc_set2:
            return 0.0
            
        intersection = len(doc_set1 & doc_set2)
        union = len(doc_set1 | doc_set2)
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_co_occurrence(
        self,
        snippets: List[Dict[str, Any]],
        term: str,
        window_size: int = 50
    ) -> Dict[str, int]:
        """
        Analyze co-occurring keywords around a term in snippets.
        
        Args:
            snippets: List of snippet dictionaries
            term: Term to analyze around
            window_size: Context window size
            
        Returns:
            Count of co-occurring keywords
        """
        # Define keyword categories
        keyword_categories = {
            'crime': ['crime', 'criminal', 'illegal', 'law', 'police', 'arrest', 'prison'],
            'policy': ['policy', 'legislation', 'law', 'reform', 'act', 'bill', 'congress'],
            'economy': ['economy', 'jobs', 'wages', 'cost', 'tax', 'business', 'market'],
            'security': ['security', 'border', 'threat', 'safety', 'protect', 'defense'],
            'social': ['family', 'community', 'society', 'culture', 'values', 'tradition'],
            'rights': ['rights', 'freedom', 'justice', 'equality', 'discrimination', 'fair']
        }
        
        co_occurrence_counts = defaultdict(int)
        
        for snippet in snippets:
            text = snippet.get('snippet_text', '').lower()
            
            # Find term occurrences
            term_lower = term.lower()
            for match in re.finditer(r'\b' + re.escape(term_lower) + r'\b', text):
                # Extract context window
                start = max(0, match.start() - window_size)
                end = min(len(text), match.end() + window_size)
                context = text[start:end]
                
                # Count keyword occurrences
                for category, keywords in keyword_categories.items():
                    for keyword in keywords:
                        if keyword in context and keyword != term_lower:
                            co_occurrence_counts[f"{category}:{keyword}"] += 1
                            
        return dict(co_occurrence_counts)
    
    def compute_bias_metrics(
        self,
        positive_outputs: List[Dict[str, Any]],
        negative_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute comparative bias metrics between positive and negative framings.
        
        Args:
            positive_outputs: Outputs from positive/neutral framings
            negative_outputs: Outputs from negative framings
            
        Returns:
            Dict of comparative metrics
        """
        # Extract sentiment scores
        pos_sentiments = [o['sentiment'] for o in positive_outputs if 'sentiment' in o]
        neg_sentiments = [o['sentiment'] for o in negative_outputs if 'sentiment' in o]
        
        # Extract toxicity scores
        pos_toxicity = [o['toxicity'] for o in positive_outputs if 'toxicity' in o]
        neg_toxicity = [o['toxicity'] for o in negative_outputs if 'toxicity' in o]
        
        # Compute differences
        metrics = {
            'sentiment_diff': np.mean(pos_sentiments) - np.mean(neg_sentiments) if pos_sentiments and neg_sentiments else 0.0,
            'toxicity_diff': np.mean(neg_toxicity) - np.mean(pos_toxicity) if pos_toxicity and neg_toxicity else 0.0,
            'pos_sentiment_mean': np.mean(pos_sentiments) if pos_sentiments else 0.0,
            'neg_sentiment_mean': np.mean(neg_sentiments) if neg_sentiments else 0.0,
            'pos_toxicity_mean': np.mean(pos_toxicity) if pos_toxicity else 0.0,
            'neg_toxicity_mean': np.mean(neg_toxicity) if neg_toxicity else 0.0,
        }
        
        return metrics