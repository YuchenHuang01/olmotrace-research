# Technical Methods

This document describes the technical approach used in the OLMo Framing Bias Audit Tool.

## Span Scoring Algorithm

### Overview

The span selection algorithm identifies important text spans in model outputs that may reveal framing-specific patterns. We use a weighted combination of three signals:

```
score(span) = w_r * rarity(span) + w_p * pmi(span) + w_a * ablation(span)
```

### 1. Rarity Score

The rarity score measures how uncommon a span's tokens are in the background corpus:

```python
rarity(span) = (1/|span|) * Σ -log P(token)
```

Where:
- `P(token)` is the unigram probability from the background corpus
- Background corpus is built from all generated outputs
- Add-one smoothing is applied to handle unseen tokens

**Rationale**: Rare phrases often indicate framing-specific language patterns.

### 2. PMI (Pointwise Mutual Information) Score

PMI measures the association strength between tokens in a span:

```python
pmi(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))
```

We calculate:
- Bigram PMI for adjacent token pairs
- Trigram PMI for three-token sequences
- Final score is the average positive PMI across all n-grams

**Rationale**: High PMI indicates idiomatic phrases characteristic of specific framings.

### 3. Ablation Score

The ablation score measures semantic drift when the framing term is replaced:

```python
ablation(span) = 1 - similarity(output_original, output_ablated)
```

Where:
- `output_ablated` is generated with neutral term replacement
- Similarity is computed using TF-IDF cosine similarity or ROUGE

**Rationale**: Spans causing high drift are likely framing-sensitive.

### Span Filtering

After scoring, we:
1. Filter spans exceeding the frequency threshold
2. Remove overlapping spans, keeping highest-scoring
3. Return top-K non-overlapping spans

## Infini-gram Retrieval

### API Interaction

For each selected span, we:

1. **Query multiple indexes**:
   - `v4_olmo-mix-1124_llama`: Pretaining data
   - `v4_olmo-2-1124-13b-instruct_llama`: Instruction data

2. **Sample document pointers**:
   ```python
   find(index, span_text) -> [(shard, rank), ...]
   ```

3. **Fetch snippets**:
   ```python
   get_doc_by_rank(index, shard, rank) -> snippet_text
   ```

### Reranking

Retrieved snippets are reranked by relevance:
- Compute TF-IDF vectors for snippets and full output
- Rank by cosine similarity
- Keep top 3-5 most relevant snippets

## Source Domain Classification

### Pattern-Based Classification

Snippets are classified into domains using regex patterns:

```python
domain_patterns = {
    'wikipedia': [r'wikipedia\.org', r'encyclop[ae]dia'],
    'mainstream_news': [r'(nytimes|washingtonpost|reuters)'],
    'gov_ngo': [r'\.gov\b', r'department|agency'],
    'blog': [r'blog|wordpress|medium\.com'],
    'forum': [r'reddit|twitter|forum|comment']
}
```

### Heuristics

1. Check for URL patterns in snippet
2. Look for domain-specific keywords
3. Default to 'other' if no patterns match

## Bias Metrics

### Output-Level Metrics

1. **Sentiment Analysis**:
   - Primary: HuggingFace sentiment model
   - Fallback: TextBlob polarity
   - Range: -1 (negative) to +1 (positive)

2. **Toxicity Detection**:
   - Pattern-based heuristics
   - Keywords: hate, threat, attack, etc.
   - Intensity markers: excessive punctuation, all-caps

3. **Subjectivity**:
   - TextBlob subjectivity score
   - Range: 0 (objective) to 1 (subjective)

### Comparative Analysis

```python
bias_score = mean(positive_framing) - mean(negative_framing)
```

Computed for:
- Sentiment difference
- Toxicity difference
- Length variation

## Statistical Significance

### Sample Size Calculation

For each cell (domain × template × framing):
- Default: 3 samples
- Adjustable via `samples_per_cell` parameter

### Aggregation

Metrics are aggregated with:
- Mean and standard deviation
- Box plots showing distribution
- Statistical tests (if sample size permits)

## Visualization Methods

### 1. Source Distribution

Grouped bar chart showing:
- X-axis: Source domains
- Y-axis: Snippet counts
- Groups: Framing types

### 2. Sentiment/Toxicity

Box plots displaying:
- Distribution per framing
- Median, quartiles, outliers
- Reference lines (zero sentiment, toxicity threshold)

### 3. Document Overlap

Heatmap of Jaccard similarity:
```python
jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Where A, B are sets of (index, doc_ptr) tuples.

### 4. Co-occurrence Analysis

Top co-occurring terms within context windows:
- Window size: 50 tokens around framing term
- Categories: crime, policy, economy, etc.
- Visualization: Grouped bar chart

## Performance Considerations

### Efficiency Optimizations

1. **Batch Processing**:
   - Group API calls by span
   - Process multiple outputs in parallel

2. **Caching**:
   - Cache background frequencies
   - Reuse TF-IDF vectorizers

3. **Rate Limiting**:
   - Configurable delays between API calls
   - Exponential backoff on failures

### Scalability

- Streaming processing for large datasets
- Configurable span limits per output
- Sampling strategies for high-frequency spans