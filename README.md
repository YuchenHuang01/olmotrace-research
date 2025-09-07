# OLMo Framing Bias Audit Tool

A production-quality tool for auditing framing bias in OLMo models by tracing important output spans back to training data using the Infini-gram API.

## Overview

This tool analyzes how different framings (e.g., "undocumented immigrants" vs "illegal aliens") affect OLMo model outputs by:

1. **Generating outputs** with paired framings across multiple prompt templates
2. **Selecting important spans** using rarity, PMI, and ablation scoring
3. **Tracing spans** back to training data via the Infini-gram API
4. **Analyzing bias metrics** including sentiment, toxicity, and source domain distribution
5. **Producing visualizations** and a comprehensive HTML report

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd framing_audit

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

1. **Configure your framings** in `config/framings.yaml`:
```yaml
immigration:
  positive_neutral: ["undocumented immigrants", "migrant workers"]
  negative: ["illegal aliens", "alien invaders"]
```

2. **Configure prompt templates** in `config/prompts.yaml`:
```yaml
neutral: "Explain the impact of {TERM} on the economy."
provocative: "Why is {TERM} a threat to our society?"
```

3. **Set run parameters** in `config/run_config.yaml`:
```yaml
model:
  name: "allenai/OLMo-2-1124-7B"
  device: "cuda"
generation:
  samples_per_cell: 3
  temperature: 0.7
```

4. **Generate framings and prompts** (optional):
```bash
# Generate framing pairs using LLM
python -m framing_audit.cli gen framings \
  --domains immigration taxation climate \
  --model-name allenai/OLMo-2-1124-13B-Instruct \
  --out config/framings.yaml

# Generate prompt templates  
python -m framing_audit.cli gen prompts --out config/prompts.yaml

# Generate prompt grid CSV for experiments
python -m framing_audit.cli gen promptgrid \
  --framings config/framings.yaml \
  --prompts config/prompts.yaml \
  --run-id my-experiment \
  --paraphrase --seed 42
```

5. **Run the audit**:
```bash
python -m framing_audit.cli audit \
  --framings config/framings.yaml \
  --prompts config/prompts.yaml \
  --run-config config/run_config.yaml \
  --run-id my-audit-2025-09
```

## Configuration

### Framings Configuration

Define framing pairs grouped by domain:
- `positive_neutral`: Terms with positive or neutral connotations
- `negative`: Terms with negative connotations

### Prompt Templates

Create templates with `{TERM}` placeholder for framing terms. Examples:
- Neutral/factual prompts
- Provocative prompts
- Balanced prompts

### Run Configuration

Key parameters:
- `model`: OLMo model selection and device settings
- `generation`: Decoding parameters (temperature, top_p, max_tokens)
- `span_selection`: Span scoring weights and thresholds
- `infinigram`: API settings and index selection
- `analysis`: Sentiment/toxicity model configuration

## Generate Framings & Prompts

The tool can automatically generate framing pairs and prompt templates using HuggingFace LLMs:

### Generate Framing Pairs

```bash
# Basic LLM-only generation
python -m framing_audit.cli gen framings \
  --domains immigration taxation climate \
  --model-name allenai/OLMo-2-1124-13B-Instruct \
  --out config/framings.yaml

# With embedding expansion and deduplication
python -m framing_audit.cli gen framings \
  --domains immigration taxation climate healthcare \
  --model-name allenai/OLMo-2-1124-13B-Instruct \
  --use-embeddings \
  --neighbors 20 \
  --max-per-bucket 6 \
  --out config/framings.yaml \
  --seed 42
```

**Options:**
- `--domains`: Space-separated domain names (required)
- `--model-name`: HuggingFace model for generation (default: allenai/OLMo-2-1124-13B-Instruct)
- `--use-embeddings`: Enable semantic expansion and deduplication
- `--neighbors`: Number of variants to generate per seed phrase (default: 20)
- `--max-per-bucket`: Maximum phrases per framing bucket (default: 6) 
- `--seed`: Random seed for reproducibility (default: 42)

The generation process:
1. **Initial Generation**: LLM creates framing terms in three buckets per domain
2. **Expansion** (if `--use-embeddings`): Generate variants of seed phrases
3. **Deduplication**: Remove near-duplicates using text similarity and embeddings
4. **Validation**: Check that phrases refer to the same concept
5. **Relabeling**: Correct bucket assignments using tone analysis
6. **Output**: Clean YAML with capped bucket sizes

### Generate Prompt Templates

```bash
python -m framing_audit.cli gen prompts --out config/prompts.yaml
```

Creates six prompt templates:
- **neutral**: Economic impact analysis
- **provocative**: Threat-focused framing
- **factual**: Policy debate summary
- **human_rights**: Ethical considerations
- **security**: National security implications  
- **empathy**: Compassionate perspective

### Generate Prompt Grid CSV

```bash
python -m framing_audit.cli gen promptgrid \
  --framings config/framings.yaml \
  --prompts config/prompts.yaml \
  --run-id 2025-09-07A \
  --paraphrase \
  --max-paraphrases 2 \
  --seed 42
```

Creates `runs/<run-id>/prompts.csv` with columns:
- `domain`: Domain from framings
- `term`: Specific framing term
- `bucket`: Framing bucket (positive_neutral, etc.)
- `template_key`: Prompt template name
- `style_key`: Optional style dimension
- `style_value`: Style value applied
- `variant_id`: 0 for original, 1+ for paraphrases
- `prompt_text`: Final prompt text

**Options:**
- `--paraphrase/--no-paraphrase`: Generate prompt variations
- `--max-paraphrases`: Maximum variations per prompt (default: 2)
- `--model-name`: Model for paraphrase generation

These generated files are then consumed by your existing audit pipeline.

## CLI Options

### Audit Command

```bash
python -m framing_audit.cli audit [OPTIONS]

Options:
  --framings PATH              Path to framings YAML file [required]
  --prompts PATH               Path to prompts YAML file [required]  
  --run-config PATH            Path to run configuration YAML [required]
  --run-id TEXT                Run ID (default: timestamp)
  --topk-spans INTEGER         Number of top spans per output (default: 5)
  --use-ablations              Enable ablation scoring for spans
  --indexes TEXT               Override Infini-gram indexes
  --samples-per-cell INTEGER   Samples per (domain, term, template)
```

## Output Structure

Results are saved to `runs/<run_id>/` with:

```
runs/<run_id>/
├── generations.csv      # All generated outputs
├── spans.csv           # Selected important spans
├── traces.csv          # Infini-gram training data traces
├── metrics_outputs.csv # Output-level metrics
├── metrics_sources.csv # Source domain analysis
├── *.png              # Visualizations
├── report.html        # Complete analysis report
├── metadata.json      # Run configuration
└── audit.log         # Execution log
```

## Example Output

### Sample Report Sections

1. **Executive Summary**: High-level bias findings
2. **Framing Examples**: Side-by-side output comparisons with traced sources
3. **Metrics Analysis**: Sentiment, toxicity, and subjectivity statistics
4. **Visualizations**: 
   - Source domain distribution by framing
   - Sentiment/toxicity box plots
   - Document overlap heatmap

### CSV File Formats

**generations.csv**
- `output_id`: Unique identifier
- `domain`, `frame_label`, `term`, `template`: Generation parameters
- `output_text`: Generated text

**spans.csv**
- `output_id`: Links to generation
- `span_text`: Selected span
- `rarity_score`, `pmi_score`, `ablation_score`: Individual scores
- `combined_score`: Weighted combination

**traces.csv**
- `span_text`: Traced span
- `index_name`: Infini-gram index
- `snippet_text`: Training data snippet
- `doc_ptr`, `shard`, `rank`: Document location

## Methodology

### Span Selection

Spans are scored using three signals:

1. **Rarity**: Sum of negative log unigram probabilities
2. **PMI**: Pointwise mutual information for bigrams/trigrams  
3. **Ablation**: Semantic drift when framing is neutralized

### Source Analysis

Training snippets are classified into domains:
- Wikipedia
- Mainstream news
- Government/NGO
- Blogs
- Forums
- Other

### Bias Metrics

- **Sentiment**: Positive/negative valence (-1 to 1)
- **Toxicity**: Harmful content score (0 to 1)
- **Subjectivity**: Opinion vs fact (0 to 1)

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Metrics

1. Extend `BiasAnalyzer` in `analysis.py`
2. Update `analyze_output_metrics()` method
3. Add visualization in `viz.py`

### Custom Infini-gram Indexes

Modify `config/run_config.yaml`:
```yaml
infinigram:
  indexes:
    - "v4_custom-index_llama"
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **API timeouts**: Increase `timeout` in config
3. **No spans selected**: Adjust frequency threshold

### Logging

Check `runs/<run_id>/audit.log` for detailed execution logs.

## Citation

If you use this tool in research, please cite:

```bibtex
@software{olmo_framing_audit,
  title = {OLMo Framing Bias Audit Tool},
  author = {OLMo Framing Audit Team},
  year = {2025},
  url = {https://github.com/...}
}
```

## License

MIT License - see LICENSE file for details.