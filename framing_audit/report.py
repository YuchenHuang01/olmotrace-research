"""
Report generation module for creating HTML reports.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from jinja2 import Template
import pandas as pd
import base64
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates HTML reports for framing bias analysis."""
    
    def __init__(self):
        """Initialize report generator."""
        self.template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>OLMo Framing Bias Audit Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 40px;
        }
        .example-box {
            background-color: #f0f0f0;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .span-highlight {
            background-color: #ffeb3b;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        .source-snippet {
            background-color: #e8f4f8;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            font-size: 0.9em;
            font-style: italic;
        }
        .metric-table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        .metric-table th, .metric-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .metric-table th {
            background-color: #3498db;
            color: white;
        }
        .metric-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .summary-box {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .warning {
            color: #e74c3c;
            font-weight: bold;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .footer {
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>OLMo Framing Bias Audit Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Run ID:</strong> {{ run_id }}</p>
        <p><strong>Model:</strong> {{ model_name }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            {{ executive_summary }}
        </div>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        <ul>
        {% for finding in key_findings %}
            <li>{{ finding }}</li>
        {% endfor %}
        </ul>
    </div>
    
    <div class="section">
        <h2>Framing Examples</h2>
        {% for example in examples %}
        <div class="example-box">
            <h3>{{ example.domain }} - {{ example.template }}</h3>
            <p><strong>Positive/Neutral Framing:</strong> "{{ example.positive_term }}"</p>
            <div style="margin-left: 20px;">
                <p><em>Generated Output:</em></p>
                <p>{{ example.positive_output | safe }}</p>
                {% if example.positive_source %}
                <div class="source-snippet">
                    <strong>Top Training Source:</strong> {{ example.positive_source }}
                </div>
                {% endif %}
            </div>
            
            <p><strong>Negative Framing:</strong> "{{ example.negative_term }}"</p>
            <div style="margin-left: 20px;">
                <p><em>Generated Output:</em></p>
                <p>{{ example.negative_output | safe }}</p>
                {% if example.negative_source %}
                <div class="source-snippet">
                    <strong>Top Training Source:</strong> {{ example.negative_source }}
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Metrics Analysis</h2>
        <table class="metric-table">
            <tr>
                <th>Framing</th>
                <th>Avg Sentiment</th>
                <th>Avg Toxicity</th>
                <th>Avg Subjectivity</th>
                <th>Toxic Outputs (%)</th>
            </tr>
            {% for row in metrics_summary %}
            <tr>
                <td>{{ row.framing }}</td>
                <td class="{% if row.sentiment < 0 %}warning{% else %}positive{% endif %}">
                    {{ "%.3f"|format(row.sentiment) }}
                </td>
                <td class="{% if row.toxicity > 0.5 %}warning{% endif %}">
                    {{ "%.3f"|format(row.toxicity) }}
                </td>
                <td>{{ "%.3f"|format(row.subjectivity) }}</td>
                <td class="{% if row.toxic_pct > 20 %}warning{% endif %}">
                    {{ "%.1f"|format(row.toxic_pct) }}%
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        
        {% if viz_paths.source_distribution %}
        <div class="chart-container">
            <h3>Source Domain Distribution</h3>
            <img src="{{ viz_paths.source_distribution }}" alt="Source Distribution">
        </div>
        {% endif %}
        
        {% if viz_paths.sentiment_toxicity %}
        <div class="chart-container">
            <h3>Sentiment and Toxicity by Framing</h3>
            <img src="{{ viz_paths.sentiment_toxicity }}" alt="Sentiment and Toxicity">
        </div>
        {% endif %}
        
        {% if viz_paths.jaccard_heatmap %}
        <div class="chart-container">
            <h3>Document Overlap Between Framings</h3>
            <img src="{{ viz_paths.jaccard_heatmap }}" alt="Jaccard Heatmap">
        </div>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Methodology</h2>
        <p>This audit analyzed framing bias in OLMo model outputs by:</p>
        <ol>
            <li>Generating text using paired framings (positive/neutral vs negative)</li>
            <li>Selecting important spans using rarity, PMI, and ablation scores</li>
            <li>Tracing spans back to training data via Infini-gram API</li>
            <li>Analyzing sentiment, toxicity, and source domain distribution</li>
        </ol>
    </div>
    
    <div class="footer">
        <p>Generated by OLMo Framing Audit Tool v{{ version }}</p>
        <p>Report generation completed in {{ duration }} seconds</p>
    </div>
</body>
</html>
        """)
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for embedding in HTML."""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
    
    def _highlight_spans(self, text: str, spans: List[Dict[str, Any]]) -> str:
        """Highlight important spans in text."""
        # Sort spans by start position (reverse order)
        sorted_spans = sorted(spans, key=lambda x: x['start'], reverse=True)
        
        highlighted = text
        for span in sorted_spans:
            start = span['start']
            end = span['end']
            span_text = highlighted[start:end]
            
            # Insert highlight markup
            highlighted = (
                highlighted[:start] +
                f'<span class="span-highlight">{span_text}</span>' +
                highlighted[end:]
            )
            
        return highlighted
    
    def _generate_executive_summary(
        self,
        metrics_df: pd.DataFrame,
        bias_metrics: Dict[str, float]
    ) -> str:
        """Generate executive summary text."""
        summary_parts = []
        
        # Overall bias direction
        if bias_metrics.get('sentiment_diff', 0) > 0.2:
            summary_parts.append(
                "The analysis reveals <strong>significant positive bias</strong> in outputs "
                "generated with positive/neutral framings compared to negative framings."
            )
        elif bias_metrics.get('sentiment_diff', 0) < -0.2:
            summary_parts.append(
                "The analysis reveals <strong>significant negative bias</strong> in outputs "
                "generated with negative framings compared to positive/neutral framings."
            )
        else:
            summary_parts.append(
                "The analysis shows <strong>relatively balanced</strong> sentiment "
                "across different framings."
            )
        
        # Toxicity findings
        toxic_pct_negative = (metrics_df[metrics_df['frame_label'] == 'negative']['is_toxic'].mean() * 100
                             if 'is_toxic' in metrics_df.columns else 0)
        
        if toxic_pct_negative > 30:
            summary_parts.append(
                f"<span class='warning'>Warning: {toxic_pct_negative:.1f}% of outputs "
                f"with negative framings exceeded toxicity thresholds.</span>"
            )
        
        # Sample size
        n_outputs = len(metrics_df)
        summary_parts.append(
            f"This report analyzes {n_outputs} generated outputs across "
            f"{len(metrics_df['domain'].unique())} domains and "
            f"{len(metrics_df['template'].unique())} prompt templates."
        )
        
        return " ".join(summary_parts)
    
    def _generate_key_findings(
        self,
        metrics_df: pd.DataFrame,
        source_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate list of key findings."""
        findings = []
        
        # Sentiment findings
        pos_sentiment = metrics_df[metrics_df['frame_label'] == 'positive_neutral']['sentiment'].mean()
        neg_sentiment = metrics_df[metrics_df['frame_label'] == 'negative']['sentiment'].mean()
        
        if abs(pos_sentiment - neg_sentiment) > 0.15:
            findings.append(
                f"Sentiment scores differ by {abs(pos_sentiment - neg_sentiment):.3f} points "
                f"between positive/neutral ({pos_sentiment:.3f}) and negative ({neg_sentiment:.3f}) framings."
            )
        
        # Source domain findings
        if source_analysis:
            # Find dominant sources
            for framing, domains in source_analysis.items():
                total = sum(domains.values())
                if total > 0:
                    dominant = max(domains.items(), key=lambda x: x[1])
                    pct = (dominant[1] / total) * 100
                    if pct > 40:
                        findings.append(
                            f"{framing.title()} framings draw heavily from {dominant[0]} sources ({pct:.1f}%)."
                        )
        
        # Length differences
        pos_length = metrics_df[metrics_df['frame_label'] == 'positive_neutral']['length'].mean()
        neg_length = metrics_df[metrics_df['frame_label'] == 'negative']['length'].mean()
        
        if abs(pos_length - neg_length) > 20:
            findings.append(
                f"Output length varies by {abs(pos_length - neg_length):.0f} words on average "
                f"between framings."
            )
        
        return findings
    
    def generate_report(
        self,
        run_id: str,
        config: Dict[str, Any],
        generations_df: pd.DataFrame,
        spans_df: pd.DataFrame,
        traces_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        viz_paths: Dict[str, str],
        output_path: str,
        duration: float
    ) -> str:
        """
        Generate complete HTML report.
        
        Args:
            run_id: Run identifier
            config: Run configuration
            generations_df: Generation data
            spans_df: Span selection data
            traces_df: Infini-gram trace data
            metrics_df: Metrics data
            viz_paths: Paths to visualization images
            output_path: Output path for report
            duration: Total run duration
            
        Returns:
            Path to generated report
        """
        # Prepare visualization paths (convert to base64)
        viz_data = {}
        for name, path in viz_paths.items():
            if os.path.exists(path):
                # For HTML embedding, use data URLs
                img_data = self._encode_image(path)
                viz_data[name] = f"data:image/png;base64,{img_data}"
            else:
                viz_data[name] = ""
        
        # Generate examples
        examples = self._generate_examples(
            generations_df,
            spans_df,
            traces_df,
            max_examples=3
        )
        
        # Calculate metrics summary
        metrics_summary = []
        for frame_label in ['positive_neutral', 'negative']:
            frame_data = metrics_df[metrics_df['frame_label'] == frame_label]
            
            if len(frame_data) > 0:
                metrics_summary.append({
                    'framing': frame_label.replace('_', '/').title(),
                    'sentiment': frame_data['sentiment'].mean(),
                    'toxicity': frame_data['toxicity'].mean(),
                    'subjectivity': frame_data['subjectivity'].mean(),
                    'toxic_pct': frame_data['is_toxic'].mean() * 100 if 'is_toxic' in frame_data else 0
                })
        
        # Calculate bias metrics
        bias_metrics = self._calculate_bias_metrics(metrics_df)
        
        # Analyze sources
        source_analysis = self._analyze_sources_by_framing(traces_df)
        
        # Generate summary and findings
        executive_summary = self._generate_executive_summary(metrics_df, bias_metrics)
        key_findings = self._generate_key_findings(metrics_df, source_analysis)
        
        # Render template
        html_content = self.template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=run_id,
            model_name=config.get('model', {}).get('name', 'Unknown'),
            executive_summary=executive_summary,
            key_findings=key_findings,
            examples=examples,
            metrics_summary=metrics_summary,
            viz_paths=viz_data,
            version="0.1.0",
            duration=f"{duration:.1f}"
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Report generated: {output_path}")
        return output_path
    
    def _generate_examples(
        self,
        generations_df: pd.DataFrame,
        spans_df: pd.DataFrame,
        traces_df: pd.DataFrame,
        max_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate example comparisons for report."""
        examples = []
        
        # Get unique domain/template combinations
        combinations = generations_df[['domain', 'template']].drop_duplicates().head(max_examples)
        
        for _, combo in combinations.iterrows():
            domain = combo['domain']
            template = combo['template']
            
            # Get positive and negative examples
            pos_examples = generations_df[
                (generations_df['domain'] == domain) &
                (generations_df['template'] == template) &
                (generations_df['frame_label'] == 'positive_neutral')
            ]
            
            neg_examples = generations_df[
                (generations_df['domain'] == domain) &
                (generations_df['template'] == template) &
                (generations_df['frame_label'] == 'negative')
            ]
            
            if len(pos_examples) > 0 and len(neg_examples) > 0:
                pos_example = pos_examples.iloc[0]
                neg_example = neg_examples.iloc[0]
                
                # Get spans for highlighting
                pos_spans = spans_df[spans_df['output_id'] == pos_example['output_id']].to_dict('records')
                neg_spans = spans_df[spans_df['output_id'] == neg_example['output_id']].to_dict('records')
                
                # Get top source snippets
                pos_source = self._get_top_source(traces_df, pos_example['output_id'])
                neg_source = self._get_top_source(traces_df, neg_example['output_id'])
                
                examples.append({
                    'domain': domain,
                    'template': template,
                    'positive_term': pos_example['term'],
                    'negative_term': neg_example['term'],
                    'positive_output': self._highlight_spans(pos_example['output_text'], pos_spans),
                    'negative_output': self._highlight_spans(neg_example['output_text'], neg_spans),
                    'positive_source': pos_source,
                    'negative_source': neg_source
                })
                
        return examples
    
    def _get_top_source(self, traces_df: pd.DataFrame, output_id: str) -> Optional[str]:
        """Get top source snippet for an output."""
        output_traces = traces_df[traces_df['output_id'] == output_id]
        
        if len(output_traces) > 0:
            top_trace = output_traces.iloc[0]
            return f"{top_trace['snippet_text'][:200]}... (from {top_trace['index_name']})"
        
        return None
    
    def _calculate_bias_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comparative bias metrics."""
        pos_data = metrics_df[metrics_df['frame_label'] == 'positive_neutral']
        neg_data = metrics_df[metrics_df['frame_label'] == 'negative']
        
        metrics = {}
        
        if len(pos_data) > 0 and len(neg_data) > 0:
            metrics['sentiment_diff'] = pos_data['sentiment'].mean() - neg_data['sentiment'].mean()
            metrics['toxicity_diff'] = neg_data['toxicity'].mean() - pos_data['toxicity'].mean()
            
        return metrics
    
    def _analyze_sources_by_framing(self, traces_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Analyze source domains by framing type."""
        # This would need to be implemented based on actual trace data structure
        # For now, return empty dict
        return {}