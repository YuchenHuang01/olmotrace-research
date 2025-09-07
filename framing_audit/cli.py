"""
Command-line interface for framing bias audit.
"""

import click
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import os

from .utils import (
    setup_logging, set_seed, load_yaml, save_json, save_csv, save_yaml,
    create_run_directory, save_metadata, format_prompt, ProgressTracker
)
from .framing_gen import (
    generate_framings, generate_prompts, render_prompt_grid, 
    optional_style_knobs, set_global_seed
)
from .models import OLMoGenerator, GenerationConfig
from .spans import SpanSelector
from .infinigram import InfinigramClient
from .analysis import BiasAnalyzer
from .viz import Visualizer
from .report import ReportGenerator


logger = logging.getLogger(__name__)


@click.group()
def cli():
    """OLMo Framing Bias Audit Tool"""
    pass


@cli.command()
@click.option('--framings', required=True, help='Path to framings YAML file')
@click.option('--prompts', required=True, help='Path to prompts YAML file')
@click.option('--run-config', required=True, help='Path to run configuration YAML file')
@click.option('--run-id', default=None, help='Run ID (default: timestamp)')
@click.option('--topk-spans', default=5, help='Number of top spans to select per output')
@click.option('--use-ablations', is_flag=True, help='Use ablation scoring for span selection')
@click.option('--indexes', multiple=True, help='Infini-gram indexes to use (overrides config)')
@click.option('--samples-per-cell', default=None, type=int, help='Samples per (domain, term, template)')
def audit(
    framings: str,
    prompts: str,
    run_config: str,
    run_id: str,
    topk_spans: int,
    use_ablations: bool,
    indexes: List[str],
    samples_per_cell: int
):
    """Run framing bias audit for OLMo models."""
    
    start_time = time.time()
    
    # Load configurations
    framings_data = load_yaml(framings)
    prompts_data = load_yaml(prompts)
    config = load_yaml(run_config)
    
    # Override config with CLI arguments
    if samples_per_cell is not None:
        config['generation']['samples_per_cell'] = samples_per_cell
    if indexes:
        config['infinigram']['indexes'] = list(indexes)
    config['span_selection']['use_ablation'] = use_ablations
    config['span_selection']['top_k_spans'] = topk_spans
    
    # Create run directory
    run_dir = create_run_directory(config['output']['base_dir'], run_id)
    
    # Set up logging
    log_file = os.path.join(run_dir, 'audit.log')
    setup_logging(log_level="INFO", log_file=log_file)
    
    logger.info(f"Starting framing bias audit - Run ID: {run_id or 'auto'}")
    logger.info(f"Output directory: {run_dir}")
    
    # Save metadata
    save_metadata(run_dir, config, {
        'framings_file': framings,
        'prompts_file': prompts,
        'cli_args': {
            'topk_spans': topk_spans,
            'use_ablations': use_ablations,
            'indexes': indexes,
            'samples_per_cell': samples_per_cell
        }
    })
    
    # Set random seed
    set_seed(config['generation']['seed'])
    
    try:
        # Step 1: Generate outputs
        logger.info("Step 1: Generating outputs with OLMo model")
        generations_df = generate_outputs(framings_data, prompts_data, config, run_dir)
        save_csv(generations_df, os.path.join(run_dir, 'generations.csv'))
        
        # Step 2: Select important spans
        logger.info("Step 2: Selecting important spans")
        spans_df = select_spans(generations_df, config, run_dir)
        save_csv(spans_df, os.path.join(run_dir, 'spans.csv'))
        
        # Step 3: Retrieve training data traces
        logger.info("Step 3: Retrieving training data traces via Infini-gram")
        traces_df = retrieve_traces(spans_df, config, run_dir)
        save_csv(traces_df, os.path.join(run_dir, 'traces.csv'))
        
        # Step 4: Analyze bias and metrics
        logger.info("Step 4: Analyzing bias metrics")
        metrics_outputs_df, metrics_sources_df = analyze_metrics(
            generations_df, spans_df, traces_df, config, run_dir
        )
        save_csv(metrics_outputs_df, os.path.join(run_dir, 'metrics_outputs.csv'))
        save_csv(metrics_sources_df, os.path.join(run_dir, 'metrics_sources.csv'))
        
        # Step 5: Create visualizations
        logger.info("Step 5: Creating visualizations")
        viz_paths = create_visualizations(
            metrics_outputs_df, metrics_sources_df, traces_df, run_dir
        )
        
        # Step 6: Generate report
        logger.info("Step 6: Generating HTML report")
        duration = time.time() - start_time
        
        report_path = generate_report(
            run_dir, config, generations_df, spans_df, traces_df,
            metrics_outputs_df, viz_paths, duration
        )
        
        logger.info(f"Audit completed in {duration:.1f} seconds")
        logger.info(f"Results saved to: {run_dir}")
        logger.info(f"Report available at: {report_path}")
        
    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        raise


def generate_outputs(
    framings_data: Dict[str, Any],
    prompts_data: Dict[str, Any],
    config: Dict[str, Any],
    run_dir: str
) -> pd.DataFrame:
    """Generate outputs for all framing/prompt combinations."""
    
    # Initialize generator
    generator = OLMoGenerator(
        model_name=config['model']['name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    
    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=config['generation']['max_new_tokens'],
        temperature=config['generation']['temperature'],
        top_p=config['generation']['top_p'],
        do_sample=config['generation']['do_sample'],
        seed=config['generation']['seed']
    )
    
    # Prepare all generation tasks
    tasks = []
    for domain, framings in framings_data.items():
        for frame_label, terms in framings.items():
            for term in terms:
                for template_name, template in prompts_data.items():
                    prompt = format_prompt(template, term)
                    tasks.append({
                        'domain': domain,
                        'frame_label': frame_label,
                        'term': term,
                        'template': template_name,
                        'prompt': prompt
                    })
    
    # Generate outputs
    results = []
    progress = ProgressTracker(len(tasks), "Generating outputs")
    
    for i, task in enumerate(tasks):
        # Update seed for this task
        task_config = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            do_sample=gen_config.do_sample,
            seed=gen_config.seed + i * 1000
        )
        
        # Generate samples
        outputs = generator.generate(
            task['prompt'],
            task_config,
            config['generation']['samples_per_cell']
        )
        
        # Store results
        for j, output_text in enumerate(outputs):
            output_id = f"{task['domain']}_{task['frame_label']}_{task['template']}_{i}_{j}"
            results.append({
                'output_id': output_id,
                'domain': task['domain'],
                'frame_label': task['frame_label'],
                'term': task['term'],
                'template': task['template'],
                'seed': task_config.seed + j,
                'prompt': task['prompt'],
                'output_text': output_text
            })
        
        progress.update()
    
    progress.finish()
    
    return pd.DataFrame(results)


def select_spans(
    generations_df: pd.DataFrame,
    config: Dict[str, Any],
    run_dir: str
) -> pd.DataFrame:
    """Select important spans from generated outputs."""
    
    # Initialize span selector
    selector = SpanSelector(
        min_span_length=config['span_selection']['min_span_length'],
        max_span_length=config['span_selection']['max_span_length'],
        rarity_weight=config['span_selection']['rarity_weight'],
        pmi_weight=config['span_selection']['pmi_weight'],
        ablation_weight=config['span_selection']['ablation_weight'],
        frequency_threshold=config['span_selection']['frequency_threshold']
    )
    
    # Build background frequencies from all outputs
    all_texts = generations_df['output_text'].tolist()
    selector.build_background_frequencies(all_texts)
    
    # Select spans for each output
    results = []
    progress = ProgressTracker(len(generations_df), "Selecting spans")
    
    for _, row in generations_df.iterrows():
        output_id = row['output_id']
        text = row['output_text']
        
        # Get ablated text if using ablation
        ablated_text = None
        if config['span_selection']['use_ablation']:
            # Generate neutral replacement
            # This is simplified - in practice, you'd regenerate with neutral term
            ablated_text = text.replace(row['term'], "[TERM]")
        
        # Select spans
        spans = selector.select_spans(
            text,
            config['span_selection']['top_k_spans'],
            ablated_text
        )
        
        # Store results
        for span in spans:
            results.append({
                'output_id': output_id,
                'span_text': span.text,
                'start': span.start,
                'end': span.end,
                'rarity_score': span.rarity_score,
                'pmi_score': span.pmi_score,
                'ablation_score': span.ablation_score,
                'combined_score': span.combined_score
            })
        
        progress.update()
    
    progress.finish()
    
    return pd.DataFrame(results)


def retrieve_traces(
    spans_df: pd.DataFrame,
    config: Dict[str, Any],
    run_dir: str
) -> pd.DataFrame:
    """Retrieve training data traces for selected spans."""
    
    # Initialize Infini-gram client
    client = InfinigramClient(
        api_url=config['infinigram']['api_url'],
        timeout=config['infinigram']['timeout'],
        retry_attempts=config['infinigram']['retry_attempts'],
        retry_delay=config['infinigram']['retry_delay']
    )
    
    # Get unique spans
    unique_spans = spans_df['span_text'].unique()
    
    # Retrieve snippets for each span
    results = []
    progress = ProgressTracker(len(unique_spans), "Retrieving traces")
    
    span_to_snippets = client.batch_retrieve(
        unique_spans.tolist(),
        config['infinigram']['indexes'],
        config['infinigram']['max_results_per_query'],
        config['infinigram']['samples_per_span']
    )
    
    # Map snippets back to outputs
    for _, row in spans_df.iterrows():
        span_text = row['span_text']
        snippets = span_to_snippets.get(span_text, [])
        
        for i, snippet in enumerate(snippets):
            results.append({
                'output_id': row['output_id'],
                'span_text': span_text,
                'index_name': snippet.index_name,
                'doc_ptr': snippet.doc_ptr,
                'shard': snippet.shard,
                'rank': snippet.rank,
                'snippet_text': snippet.snippet_text,
                'offset_left': snippet.offset_left,
                'offset_right': snippet.offset_right,
                'relevance_rank': i
            })
        
        progress.update()
    
    progress.finish()
    
    return pd.DataFrame(results)


def analyze_metrics(
    generations_df: pd.DataFrame,
    spans_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    config: Dict[str, Any],
    run_dir: str
) -> tuple:
    """Analyze bias metrics for outputs and sources."""
    
    # Initialize analyzer
    analyzer = BiasAnalyzer(
        sentiment_model=config['analysis']['sentiment_model'],
        toxicity_threshold=config['analysis']['toxicity_threshold']
    )
    
    # Analyze output metrics
    output_metrics = []
    progress = ProgressTracker(len(generations_df), "Analyzing outputs")
    
    for _, row in generations_df.iterrows():
        metrics = analyzer.analyze_output_metrics(row['output_text'])
        metrics['output_id'] = row['output_id']
        metrics['domain'] = row['domain']
        metrics['frame_label'] = row['frame_label']
        metrics['template'] = row['template']
        
        # Add coverage metric (fraction of output covered by selected spans)
        output_spans = spans_df[spans_df['output_id'] == row['output_id']]
        if len(output_spans) > 0:
            total_span_length = sum(s['end'] - s['start'] for _, s in output_spans.iterrows())
            metrics['coverage'] = total_span_length / len(row['output_text'])
        else:
            metrics['coverage'] = 0.0
        
        output_metrics.append(metrics)
        progress.update()
    
    progress.finish()
    
    # Analyze source metrics
    source_metrics = []
    
    # Group traces by span
    for span_text, group in traces_df.groupby('span_text'):
        # Analyze domain distribution
        snippets = group.to_dict('records')
        domain_dist = analyzer.analyze_source_distribution(snippets)
        
        # Get associated outputs
        span_outputs = spans_df[spans_df['span_text'] == span_text]['output_id'].unique()
        output_info = generations_df[generations_df['output_id'].isin(span_outputs)]
        
        for index_name in group['index_name'].unique():
            index_snippets = group[group['index_name'] == index_name]
            
            for domain, count in domain_dist.items():
                source_metrics.append({
                    'span_text': span_text,
                    'index_name': index_name,
                    'domain_label': domain,
                    'source_bucket': domain,
                    'count': count,
                    'frame_labels': ','.join(output_info['frame_label'].unique()),
                    'domains': ','.join(output_info['domain'].unique())
                })
    
    metrics_outputs_df = pd.DataFrame(output_metrics)
    metrics_sources_df = pd.DataFrame(source_metrics)
    
    return metrics_outputs_df, metrics_sources_df


def create_visualizations(
    metrics_outputs_df: pd.DataFrame,
    metrics_sources_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    run_dir: str
) -> Dict[str, str]:
    """Create all visualizations."""
    
    visualizer = Visualizer(run_dir)
    
    # Prepare source distribution data
    source_data = {}
    for frame_label in metrics_outputs_df['frame_label'].unique():
        # Get traces for this framing
        frame_outputs = metrics_outputs_df[
            metrics_outputs_df['frame_label'] == frame_label
        ]['output_id'].unique()
        
        frame_traces = traces_df[traces_df['output_id'].isin(frame_outputs)]
        
        # Count domains
        domain_counts = {}
        if 'domain_label' in metrics_sources_df.columns:
            frame_sources = metrics_sources_df[
                metrics_sources_df['frame_labels'].str.contains(frame_label)
            ]
            domain_counts = frame_sources.groupby('domain_label')['count'].sum().to_dict()
        
        source_data[frame_label] = domain_counts
    
    # Create Jaccard matrix (simplified)
    frame_labels = metrics_outputs_df['frame_label'].unique()
    jaccard_matrix = pd.DataFrame(
        index=frame_labels,
        columns=frame_labels,
        data=1.0  # Placeholder - would calculate actual Jaccard
    )
    
    # Create co-occurrence data (simplified)
    cooccurrence_data = {
        frame_label: {'policy:reform': 10, 'economy:jobs': 8}
        for frame_label in frame_labels
    }
    
    # Generate all visualizations
    viz_paths = visualizer.create_all_visualizations(
        source_data,
        metrics_outputs_df,
        jaccard_matrix,
        cooccurrence_data
    )
    
    return viz_paths


def generate_report(
    run_dir: str,
    config: Dict[str, Any],
    generations_df: pd.DataFrame,
    spans_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    metrics_outputs_df: pd.DataFrame,
    viz_paths: Dict[str, str],
    duration: float
) -> str:
    """Generate final HTML report."""
    
    report_gen = ReportGenerator()
    
    report_path = os.path.join(run_dir, 'report.html')
    
    report_gen.generate_report(
        run_id=os.path.basename(run_dir),
        config=config,
        generations_df=generations_df,
        spans_df=spans_df,
        traces_df=traces_df,
        metrics_df=metrics_outputs_df,
        viz_paths=viz_paths,
        output_path=report_path,
        duration=duration
    )
    
    return report_path


# Generation subcommands
@cli.group()
def gen():
    """Generate framing pairs and prompt templates."""
    pass


@gen.command()
@click.option('--domains', multiple=True, required=True, 
              help='Domains to generate framings for (e.g., immigration, taxation)')
@click.option('--model-name', default='allenai/OLMo-2-1124-13B-Instruct',
              help='HuggingFace model name for generation')
@click.option('--use-embeddings/--no-embeddings', default=False,
              help='Use sentence embeddings for expansion and deduplication')
@click.option('--neighbors', default=20, type=int,
              help='Number of variants to generate per seed phrase')
@click.option('--max-per-bucket', default=6, type=int,
              help='Maximum phrases per bucket')
@click.option('--out', default='config/framings.yaml',
              help='Output path for framings YAML')
@click.option('--seed', default=42, type=int,
              help='Random seed for reproducibility')
@click.option('--aligned/--no-aligned', default=True,
              help='Generate aligned framing sets (default) or independent buckets')
def framings(
    domains: List[str],
    model_name: str,
    use_embeddings: bool,
    neighbors: int,
    max_per_bucket: int,
    out: str,
    seed: int,
    aligned: bool
):
    """Generate framing pairs using LLM."""
    
    # Set up logging
    setup_logging(log_level="INFO")
    logger.info(f"Generating framings for domains: {list(domains)}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Use embeddings: {use_embeddings}")
    
    try:
        # Generate framings
        framings_data = generate_framings(
            domains=list(domains),
            model_name=model_name,
            use_embeddings=use_embeddings,
            max_per_bucket=max_per_bucket,
            seed=seed,
            aligned=aligned
        )
        
        # Save to YAML
        save_yaml(framings_data, out)
        
        # Print summary
        print("\n=== Generation Summary ===")
        total_phrases = 0
        for domain, domain_data in framings_data.items():
            print(f"\n{domain}:")
            if isinstance(domain_data, list):
                # Aligned format
                print(f"  Generated {len(domain_data)} aligned sets:")
                for idx, framing_set in enumerate(domain_data):
                    print(f"  Set {idx + 1}:")
                    for bucket, phrase in framing_set.items():
                        if phrase:
                            total_phrases += 1
                            print(f"    {bucket}: {phrase}")
            else:
                # Bucket format
                for bucket_name, phrases in domain_data.items():
                    count = len(phrases)
                    total_phrases += count
                    print(f"  {bucket_name}: {count} phrases")
                    for phrase in phrases:
                        print(f"    - {phrase}")
        
        print(f"\nTotal phrases generated: {total_phrases}")
        print(f"Results saved to: {out}")
        
    except Exception as e:
        logger.error(f"Failed to generate framings: {e}", exc_info=True)
        raise click.ClickException(str(e))


@gen.command()
@click.option('--out', default='config/prompts.yaml',
              help='Output path for prompts YAML')
def prompts(out: str):
    """Generate prompt templates."""
    
    setup_logging(log_level="INFO")
    logger.info("Generating prompt templates")
    
    try:
        # Generate prompts
        prompts_data = generate_prompts()
        
        # Also save style knobs
        style_data = optional_style_knobs()
        
        # Combine into full config
        full_config = {
            'templates': prompts_data,
            'styles': style_data
        }
        
        # Save to YAML
        save_yaml(full_config, out)
        
        # Print summary
        print("\n=== Prompt Templates Generated ===")
        for template_name, template in prompts_data.items():
            print(f"{template_name}: {template}")
        
        print(f"\nResults saved to: {out}")
        
    except Exception as e:
        logger.error(f"Failed to generate prompts: {e}", exc_info=True)
        raise click.ClickException(str(e))


@gen.command()
@click.option('--framings', required=True, help='Path to framings YAML file')
@click.option('--prompts', required=True, help='Path to prompts YAML file')
@click.option('--run-id', required=True, help='Run ID for output directory')
@click.option('--paraphrase/--no-paraphrase', default=False,
              help='Generate paraphrases of prompts')
@click.option('--max-paraphrases', default=2, type=int,
              help='Maximum paraphrases per prompt')
@click.option('--seed', default=42, type=int,
              help='Random seed for reproducibility')
@click.option('--model-name', default='allenai/OLMo-2-1124-13B-Instruct',
              help='Model for paraphrase generation')
def promptgrid(
    framings: str,
    prompts: str,
    run_id: str,
    paraphrase: bool,
    max_paraphrases: int,
    seed: int,
    model_name: str
):
    """Generate prompt grid CSV from framings and templates."""
    
    setup_logging(log_level="INFO")
    logger.info(f"Generating prompt grid for run: {run_id}")
    
    try:
        # Load configurations
        framings_data = load_yaml(framings)
        prompts_config = load_yaml(prompts)
        
        # Extract templates and styles
        if isinstance(prompts_config, dict) and 'templates' in prompts_config:
            prompts_data = prompts_config['templates']
            styles_data = prompts_config.get('styles', None)
        else:
            # Backward compatibility
            prompts_data = prompts_config
            styles_data = None
        
        # Generate prompt grid
        csv_path = render_prompt_grid(
            framings=framings_data,
            prompts=prompts_data,
            styles=styles_data,
            run_id=run_id,
            paraphrase=paraphrase,
            max_paraphrases=max_paraphrases,
            seed=seed,
            model_name=model_name
        )
        
        # Print summary
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print(f"\n=== Prompt Grid Summary ===")
        print(f"Total prompts: {len(df)}")
        print(f"Domains: {df['domain'].nunique()}")
        print(f"Templates: {df['template_key'].nunique()}")
        print(f"Terms: {df['term'].nunique()}")
        
        if paraphrase:
            print(f"Paraphrase variants: {df[df['variant_id'] > 0].shape[0]}")
            
        print(f"\nResults saved to: {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate prompt grid: {e}", exc_info=True)
        raise click.ClickException(str(e))


def main():
    """Main entry point - keep for backward compatibility."""
    cli()


if __name__ == "__main__":
    cli()