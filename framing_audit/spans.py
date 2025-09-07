"""
Span selection module with rarity/PMI/ablation scoring.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a text span with scoring information."""
    text: str
    start: int
    end: int
    rarity_score: float = 0.0
    pmi_score: float = 0.0
    ablation_score: float = 0.0
    combined_score: float = 0.0
    

class SpanSelector:
    """Selects important spans using rarity, PMI, and ablation signals."""
    
    def __init__(
        self,
        min_span_length: int = 3,
        max_span_length: int = 10,
        rarity_weight: float = 0.5,
        pmi_weight: float = 0.3,
        ablation_weight: float = 0.2,
        frequency_threshold: float = 0.01,
    ):
        """
        Initialize span selector.
        
        Args:
            min_span_length: Minimum span length in tokens
            max_span_length: Maximum span length in tokens
            rarity_weight: Weight for rarity score
            pmi_weight: Weight for PMI score
            ablation_weight: Weight for ablation score
            frequency_threshold: Drop spans with frequency > this
        """
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.rarity_weight = rarity_weight
        self.pmi_weight = pmi_weight
        self.ablation_weight = ablation_weight
        self.frequency_threshold = frequency_threshold
        
        # Build background frequency from corpus (to be populated)
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.total_tokens = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def build_background_frequencies(self, corpus: List[str]):
        """
        Build background frequency distributions from corpus.
        
        Args:
            corpus: List of text documents
        """
        logger.info(f"Building background frequencies from {len(corpus)} documents")
        
        for doc in corpus:
            tokens = self.tokenize(doc)
            self.total_tokens += len(tokens)
            
            # Unigrams
            self.unigram_counts.update(tokens)
            
            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_counts[bigram] += 1
                
            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.trigram_counts[trigram] += 1
                
    def get_rarity_score(self, span_tokens: List[str]) -> float:
        """
        Calculate rarity score as sum of negative log probabilities.
        
        Args:
            span_tokens: List of tokens in span
            
        Returns:
            Rarity score
        """
        if self.total_tokens == 0:
            # If no background corpus, use uniform distribution
            return len(span_tokens) * np.log(1000)  # Arbitrary large vocab
            
        score = 0.0
        for token in span_tokens:
            count = self.unigram_counts.get(token, 1)  # Add-one smoothing
            prob = count / (self.total_tokens + len(self.unigram_counts))
            score += -np.log(prob)
            
        return score / len(span_tokens)  # Normalize by length
    
    def get_pmi_score(self, span_tokens: List[str]) -> float:
        """
        Calculate PMI score for bigrams and trigrams in span.
        
        Args:
            span_tokens: List of tokens in span
            
        Returns:
            PMI score
        """
        if self.total_tokens == 0 or len(span_tokens) < 2:
            return 0.0
            
        scores = []
        
        # Bigram PMI
        for i in range(len(span_tokens) - 1):
            bigram = (span_tokens[i], span_tokens[i + 1])
            joint_count = self.bigram_counts.get(bigram, 1)
            marginal1 = self.unigram_counts.get(span_tokens[i], 1)
            marginal2 = self.unigram_counts.get(span_tokens[i + 1], 1)
            
            pmi = np.log(joint_count * self.total_tokens / (marginal1 * marginal2))
            scores.append(max(0, pmi))  # Use positive PMI
            
        # Trigram PMI (if span is long enough)
        if len(span_tokens) >= 3:
            for i in range(len(span_tokens) - 2):
                trigram = (span_tokens[i], span_tokens[i + 1], span_tokens[i + 2])
                joint_count = self.trigram_counts.get(trigram, 1)
                
                # Approximate with bigram marginals
                bigram1 = (span_tokens[i], span_tokens[i + 1])
                bigram2 = (span_tokens[i + 1], span_tokens[i + 2])
                marginal1 = self.bigram_counts.get(bigram1, 1)
                marginal2 = self.bigram_counts.get(bigram2, 1)
                
                pmi = np.log(joint_count * self.total_tokens / (marginal1 * marginal2))
                scores.append(max(0, pmi))
                
        return np.mean(scores) if scores else 0.0
    
    def get_ablation_score(
        self,
        original_text: str,
        ablated_text: str,
        method: str = "cosine"
    ) -> float:
        """
        Calculate ablation score based on semantic drift.
        
        Args:
            original_text: Original generated text
            ablated_text: Text generated with neutral replacement
            method: Similarity method ('cosine' or 'rouge')
            
        Returns:
            Ablation score (higher = more drift)
        """
        if method == "cosine":
            # Use TF-IDF cosine similarity
            vectorizer = TfidfVectorizer()
            try:
                vectors = vectorizer.fit_transform([original_text, ablated_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]
                return 1 - similarity  # Convert to drift score
            except:
                return 0.0
        elif method == "rouge":
            # Simple ROUGE-1 (unigram overlap)
            orig_tokens = set(self.tokenize(original_text))
            ablated_tokens = set(self.tokenize(ablated_text))
            
            if not orig_tokens or not ablated_tokens:
                return 0.0
                
            overlap = len(orig_tokens & ablated_tokens)
            rouge_score = overlap / max(len(orig_tokens), len(ablated_tokens))
            return 1 - rouge_score
        else:
            return 0.0
    
    def extract_candidate_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract all candidate spans from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (span_text, start_idx, end_idx) tuples
        """
        tokens = self.tokenize(text)
        token_positions = []
        
        # Track token positions in original text
        current_pos = 0
        for token in tokens:
            # Find token in text (case-insensitive)
            match = re.search(re.escape(token), text[current_pos:], re.IGNORECASE)
            if match:
                start = current_pos + match.start()
                end = current_pos + match.end()
                token_positions.append((token, start, end))
                current_pos = end
                
        # Extract spans of different lengths
        spans = []
        for length in range(self.min_span_length, self.max_span_length + 1):
            for i in range(len(token_positions) - length + 1):
                span_tokens = [t[0] for t in token_positions[i:i + length]]
                span_text = " ".join(span_tokens)
                start_pos = token_positions[i][1]
                end_pos = token_positions[i + length - 1][2]
                spans.append((span_text, start_pos, end_pos))
                
        return spans
    
    def filter_overlapping_spans(self, spans: List[Span], top_k: int) -> List[Span]:
        """
        Filter overlapping spans, keeping top-scoring non-overlapping ones.
        
        Args:
            spans: List of scored spans
            top_k: Maximum number of spans to return
            
        Returns:
            List of non-overlapping spans
        """
        # Sort by combined score (descending)
        sorted_spans = sorted(spans, key=lambda x: x.combined_score, reverse=True)
        
        selected = []
        covered_positions = set()
        
        for span in sorted_spans:
            # Check if span overlaps with already selected spans
            span_positions = set(range(span.start, span.end))
            if not span_positions & covered_positions:
                selected.append(span)
                covered_positions.update(span_positions)
                
                if len(selected) >= top_k:
                    break
                    
        return selected
    
    def select_spans(
        self,
        text: str,
        top_k: int = 5,
        ablated_text: Optional[str] = None,
    ) -> List[Span]:
        """
        Select top-k important spans from text.
        
        Args:
            text: Input text
            top_k: Number of spans to select
            ablated_text: Text generated with neutral replacement (for ablation scoring)
            
        Returns:
            List of selected spans with scores
        """
        # Extract candidate spans
        candidate_spans = self.extract_candidate_spans(text)
        
        scored_spans = []
        for span_text, start, end in candidate_spans:
            # Tokenize span
            span_tokens = self.tokenize(span_text)
            
            # Calculate scores
            rarity_score = self.get_rarity_score(span_tokens)
            pmi_score = self.get_pmi_score(span_tokens)
            
            # Ablation score (if ablated text provided)
            ablation_score = 0.0
            if ablated_text and self.ablation_weight > 0:
                # Find corresponding span in ablated text
                ablation_score = self.get_ablation_score(
                    span_text,
                    ablated_text[start:end] if end <= len(ablated_text) else ""
                )
                
            # Calculate combined score
            combined_score = (
                self.rarity_weight * rarity_score +
                self.pmi_weight * pmi_score +
                self.ablation_weight * ablation_score
            )
            
            # Filter by frequency threshold
            span_freq = sum(self.unigram_counts.get(t, 0) for t in span_tokens)
            if self.total_tokens > 0:
                span_freq /= self.total_tokens
            
            if span_freq <= self.frequency_threshold:
                span = Span(
                    text=span_text,
                    start=start,
                    end=end,
                    rarity_score=rarity_score,
                    pmi_score=pmi_score,
                    ablation_score=ablation_score,
                    combined_score=combined_score
                )
                scored_spans.append(span)
                
        # Filter overlapping spans and return top-k
        return self.filter_overlapping_spans(scored_spans, top_k)