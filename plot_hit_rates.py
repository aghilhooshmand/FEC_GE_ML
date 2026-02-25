#!/usr/bin/env python3
"""Script to create Plotly charts for hit rate and fake hit rate across sample sizes."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load the CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def create_charts_for_sampling_method(df: pd.DataFrame, sampling_method: str, output_dir: Path):
    """Create hit rate and fake hit rate charts for a specific sampling method."""
    method_df = df[df['sampling_method'] == sampling_method].copy()
    
    if method_df.empty:
        print(f"No data found for sampling method: {sampling_method}")
        return
    
    # Sort by sample_fraction for proper line plotting
    method_df = method_df.sort_values('sample_fraction')
    
    # Create subplot with 2 y-axes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Hit Rate vs Sample Size', 'Fake Hit Rate vs Sample Size'),
        horizontal_spacing=0.15
    )
    
    # Chart 1: Hit Rate (Cumulative)
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['hit_rate_cumulative'],
            mode='lines+markers',
            name='Hit Rate (Cumulative)',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Hit Rate: %{y:.4f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Chart 1: Hit Rate (Avg Per Generation)
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['hit_rate_avg_per_generation'],
            mode='lines+markers',
            name='Hit Rate (Avg Per Gen)',
            line=dict(color='lightblue', width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Hit Rate (Avg): %{y:.4f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Chart 2: Fake Hit Rate (Cumulative)
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['fake_hit_rate_cumulative'],
            mode='lines+markers',
            name='Fake Hit Rate (Cumulative)',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Fake Hit Rate: %{y:.4f}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Chart 2: Fake Hit Rate (Avg Per Generation)
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['fake_hit_rate_avg_per_generation'],
            mode='lines+markers',
            name='Fake Hit Rate (Avg Per Gen)',
            line=dict(color='lightcoral', width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Fake Hit Rate (Avg): %{y:.4f}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Hit Rate", row=1, col=1)
    fig.update_yaxes(title_text="Fake Hit Rate", row=1, col=2)
    
    fig.update_layout(
        title=f'Hit Rate and Fake Hit Rate vs Sample Size - {sampling_method.upper()}',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Save chart
    output_file = output_dir / f'hit_rates_{sampling_method}.html'
    fig.write_html(str(output_file))
    print(f"Saved chart: {output_file}")


def create_fitness_charts_for_sampling_method(df: pd.DataFrame, sampling_method: str, output_dir: Path):
    """Create training and testing fitness charts for a specific sampling method."""
    method_df = df[df['sampling_method'] == sampling_method].copy()
    
    if method_df.empty:
        print(f"No data found for sampling method: {sampling_method}")
        return
    
    # Sort by sample_fraction for proper line plotting
    method_df = method_df.sort_values('sample_fraction')
    
    # Create subplot with 2 y-axes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Fitness vs Sample Size', 'Testing Fitness vs Sample Size'),
        horizontal_spacing=0.15
    )
    
    # Chart 1: Training Fitness (using avg_fitness_last_gen)
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['avg_fitness_last_gen'],
            mode='lines+markers',
            name='Training Fitness (Avg)',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Training Fitness (Avg): %{y:.4f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Chart 2: Testing Fitness
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['test_fitness_last_gen'],
            mode='lines+markers',
            name='Testing Fitness',
            line=dict(color='orange', width=2),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Testing Fitness: %{y:.4f}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Training Fitness (lower is better)", row=1, col=1)
    fig.update_yaxes(title_text="Testing Fitness (lower is better)", row=1, col=2)
    
    fig.update_layout(
        title=f'Training and Testing Fitness vs Sample Size - {sampling_method.upper()}',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Save chart
    output_file = output_dir / f'fitness_{sampling_method}.html'
    fig.write_html(str(output_file))
    print(f"Saved fitness chart: {output_file}")


def create_complexity_charts_for_sampling_method(df: pd.DataFrame, sampling_method: str, output_dir: Path):
    """Create complexity chart for a specific sampling method."""
    method_df = df[df['sampling_method'] == sampling_method].copy()
    
    if method_df.empty:
        print(f"No data found for sampling method: {sampling_method}")
        return
    
    # Sort by sample_fraction for proper line plotting
    method_df = method_df.sort_values('sample_fraction')
    
    # Create figure
    fig = go.Figure()
    
    # Complexity (avg_nodes)
    fig.add_trace(
        go.Scatter(
            x=method_df['sample_fraction'],
            y=method_df['complexity_avg_nodes'],
            mode='lines+markers',
            name='Complexity (Avg Nodes)',
            line=dict(color='purple', width=2),
            marker=dict(size=8),
            hovertemplate='Sample Fraction: %{x:.2%}<br>Complexity (Avg Nodes): %{y:.2f}<extra></extra>',
        )
    )
    
    # Update layout
    fig.update_xaxes(title_text="Sample Fraction")
    fig.update_yaxes(title_text="Complexity (Average Nodes)")
    
    fig.update_layout(
        title=f'Complexity vs Sample Size - {sampling_method.upper()}',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Save chart
    output_file = output_dir / f'complexity_{sampling_method}.html'
    fig.write_html(str(output_file))
    print(f"Saved complexity chart: {output_file}")


def create_complexity_comparison_chart(df: pd.DataFrame, output_dir: Path):
    """Create comparison chart for complexity across all sampling methods."""
    # Get unique sampling methods
    sampling_methods = df['sampling_method'].unique()
    
    if len(sampling_methods) == 0:
        print("No sampling methods found in data")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, method in enumerate(sampling_methods):
        method_df = df[df['sampling_method'] == method].copy()
        method_df = method_df.sort_values('sample_fraction')
        
        color = colors[idx % len(colors)]
        
        # Complexity
        fig.add_trace(
            go.Scatter(
                x=method_df['sample_fraction'],
                y=method_df['complexity_avg_nodes'],
                mode='lines+markers',
                name=f'{method}',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=f'Method: {method}<br>Sample Fraction: %{{x:.2%}}<br>Complexity: %{{y:.2f}}<extra></extra>',
            )
        )
    
    # Update layout
    fig.update_xaxes(title_text="Sample Fraction")
    fig.update_yaxes(title_text="Complexity (Average Nodes)")
    
    fig.update_layout(
        title='Complexity vs Sample Size - All Sampling Methods Comparison',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    # Save chart
    output_file = output_dir / 'complexity_comparison_all_methods.html'
    fig.write_html(str(output_file))
    print(f"Saved complexity comparison chart: {output_file}")


def create_fitness_comparison_chart(df: pd.DataFrame, output_dir: Path):
    """Create comparison charts for training and testing fitness across all sampling methods."""
    # Get unique sampling methods
    sampling_methods = df['sampling_method'].unique()
    
    if len(sampling_methods) == 0:
        print("No sampling methods found in data")
        return
    
    # Create subplot with 2 y-axes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Fitness vs Sample Size (All Methods)', 'Testing Fitness vs Sample Size (All Methods)'),
        horizontal_spacing=0.15
    )
    
    # Color palette for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, method in enumerate(sampling_methods):
        method_df = df[df['sampling_method'] == method].copy()
        method_df = method_df.sort_values('sample_fraction')
        
        color = colors[idx % len(colors)]
        
        # Training Fitness (using avg_fitness_last_gen)
        fig.add_trace(
            go.Scatter(
                x=method_df['sample_fraction'],
                y=method_df['avg_fitness_last_gen'],
                mode='lines+markers',
                name=f'{method} - Train',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=f'Method: {method}<br>Sample Fraction: %{{x:.2%}}<br>Training Fitness (Avg): %{{y:.4f}}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # Testing Fitness
        fig.add_trace(
            go.Scatter(
                x=method_df['sample_fraction'],
                y=method_df['test_fitness_last_gen'],
                mode='lines+markers',
                name=f'{method} - Test',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate=f'Method: {method}<br>Sample Fraction: %{{x:.2%}}<br>Testing Fitness: %{{y:.4f}}<extra></extra>',
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Training Fitness (lower is better)", row=1, col=1)
    fig.update_yaxes(title_text="Testing Fitness (lower is better)", row=1, col=2)
    
    fig.update_layout(
        title='Training and Testing Fitness vs Sample Size - All Sampling Methods Comparison',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    # Save chart
    output_file = output_dir / 'fitness_comparison_all_methods.html'
    fig.write_html(str(output_file))
    print(f"Saved fitness comparison chart: {output_file}")


def create_comparison_chart(df: pd.DataFrame, output_dir: Path):
    """Create comparison charts showing all sampling methods."""
    # Get unique sampling methods
    sampling_methods = df['sampling_method'].unique()
    
    if len(sampling_methods) == 0:
        print("No sampling methods found in data")
        return
    
    # Create subplot with 2 y-axes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Hit Rate vs Sample Size (All Methods)', 'Fake Hit Rate vs Sample Size (All Methods)'),
        horizontal_spacing=0.15
    )
    
    # Color palette for different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, method in enumerate(sampling_methods):
        method_df = df[df['sampling_method'] == method].copy()
        method_df = method_df.sort_values('sample_fraction')
        
        color = colors[idx % len(colors)]
        
        # Hit Rate (Cumulative)
        fig.add_trace(
            go.Scatter(
                x=method_df['sample_fraction'],
                y=method_df['hit_rate_cumulative'],
                mode='lines+markers',
                name=f'{method} - Hit Rate',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=f'Method: {method}<br>Sample Fraction: %{{x:.2%}}<br>Hit Rate: %{{y:.4f}}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # Fake Hit Rate (Cumulative)
        fig.add_trace(
            go.Scatter(
                x=method_df['sample_fraction'],
                y=method_df['fake_hit_rate_cumulative'],
                mode='lines+markers',
                name=f'{method} - Fake Hit Rate',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate=f'Method: {method}<br>Sample Fraction: %{{x:.2%}}<br>Fake Hit Rate: %{{y:.4f}}<extra></extra>',
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Sample Fraction", row=1, col=2)
    fig.update_yaxes(title_text="Hit Rate", row=1, col=1)
    fig.update_yaxes(title_text="Fake Hit Rate", row=1, col=2)
    
    fig.update_layout(
        title='Hit Rate and Fake Hit Rate vs Sample Size - All Sampling Methods Comparison',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    # Save chart
    output_file = output_dir / 'hit_rates_comparison_all_methods.html'
    fig.write_html(str(output_file))
    print(f"Saved comparison chart: {output_file}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python plot_hit_rates.py <csv_file_path>")
        print("Example: python plot_hit_rates.py results/run002/run002_all_experiments_20251123_152010.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Load data
    df = load_csv(str(csv_path))
    
    # Create output directory
    output_dir = csv_path.parent / 'charts'
    output_dir.mkdir(exist_ok=True)
    
    # Get unique sampling methods
    sampling_methods = df['sampling_method'].unique()
    print(f"\nFound sampling methods: {list(sampling_methods)}")
    
    # Create hit rate charts for each sampling method
    print("\nCreating hit rate charts for each sampling method...")
    for method in sampling_methods:
        create_charts_for_sampling_method(df, method, output_dir)
    
    # Create hit rate comparison chart
    print("\nCreating hit rate comparison chart for all methods...")
    create_comparison_chart(df, output_dir)
    
    # Create fitness charts for each sampling method
    print("\nCreating fitness charts for each sampling method...")
    for method in sampling_methods:
        create_fitness_charts_for_sampling_method(df, method, output_dir)
    
    # Create fitness comparison chart
    print("\nCreating fitness comparison chart for all methods...")
    create_fitness_comparison_chart(df, output_dir)
    
    # Create complexity charts for each sampling method
    print("\nCreating complexity charts for each sampling method...")
    for method in sampling_methods:
        create_complexity_charts_for_sampling_method(df, method, output_dir)
    
    # Create complexity comparison chart
    print("\nCreating complexity comparison chart for all methods...")
    create_complexity_comparison_chart(df, output_dir)
    
    print(f"\nAll charts saved to: {output_dir}")


if __name__ == "__main__":
    main()

