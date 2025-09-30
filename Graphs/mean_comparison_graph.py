#!/usr/bin/env python3
"""
CSV Statistics Comparison Tool using Seaborn
Compares means, standard deviations, and variances across multiple columns
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import numpy as np

def create_statistics_comparison(csv_file, columns, separator=',', output_file=None, 
                                stat_type='mean', figsize=(10, 6), title=None, 
                                color='skyblue', show_values=True, group_by=None):
    """
    Create a comparison visualization of statistics across multiple columns.
    
    Parameters:
    csv_file (str): Path to the CSV file
    columns (list): List of column names to compare
    separator (str): CSV separator character
    output_file (str): Path to save the plot (optional)
    stat_type (str): Type of statistic ('mean', 'std', 'variance', 'all')
    figsize (tuple): Figure size (width, height)
    title (str): Custom title for the plot
    color (str): Color scheme for the plot
    show_values (bool): Whether to display values on bars
    group_by (str): Column name to group by (optional)
    """
    
    try:
        df = pd.read_csv(csv_file, sep=separator)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        
        # Check if columns exist
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Columns {missing_columns} not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        if group_by and group_by not in df.columns:
            print(f"Error: Group column '{group_by}' not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Validate that columns are numeric
        non_numeric = []
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"Converted column '{col}' to numeric.")
                except:
                    non_numeric.append(col)
        
        if non_numeric:
            print(f"Error: Could not convert columns {non_numeric} to numeric values.")
            return False
        
        # Remove rows with NaN values in specified columns
        df_clean = df.dropna(subset=columns)
        if len(df_clean) == 0:
            print(f"Error: No valid data found after removing NaN values.")
            return False
        
        if len(df_clean) < len(df):
            print(f"Removed {len(df) - len(df_clean)} rows with missing values.")
        
        sns.set_style("whitegrid")
        
        if stat_type == 'all':
            # Create subplots for mean, std, and variance
            fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))
            
            if group_by:
                stats_data = _calculate_grouped_statistics(df_clean, columns, group_by)
                _plot_grouped_statistics(axes, stats_data, columns, group_by, color, show_values)
            else:
                stats_data = _calculate_statistics(df_clean, columns)
                _plot_all_statistics(axes, stats_data, columns, color, show_values)
            
            if title:
                fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
            else:
                fig.suptitle('Statistics Comparison', fontsize=18, fontweight='bold', y=1.02)
            
            plt.tight_layout()
        else:
            # Create single plot
            plt.figure(figsize=figsize)
            
            if group_by:
                stats_data = _calculate_grouped_statistics(df_clean, columns, group_by)
                _plot_single_grouped_statistic(stats_data, columns, group_by, stat_type, 
                                               color, show_values)
            else:
                stats_data = _calculate_statistics(df_clean, columns)
                _plot_single_statistic(stats_data, columns, stat_type, color, show_values)
            
            if title is None:
                stat_labels = {
                    'mean': 'Mean',
                    'std': 'Standard Deviation',
                    'variance': 'Variance'
                }
                title = f"{stat_labels.get(stat_type, stat_type.title())} Comparison"
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved as '{output_file}'")
        else:
            plt.show()
        
        # Print statistics table
        _print_statistics_table(df_clean, columns, group_by)
        
        return True
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_file}' is empty.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return False

def _calculate_statistics(df, columns):
    """Calculate mean, std, and variance for specified columns."""
    stats = {
        'mean': [],
        'std': [],
        'variance': []
    }
    
    for col in columns:
        stats['mean'].append(df[col].mean())
        stats['std'].append(df[col].std())
        stats['variance'].append(df[col].var())
    
    return stats

def _calculate_grouped_statistics(df, columns, group_by):
    """Calculate statistics grouped by a categorical column."""
    grouped_stats = {}
    
    for group in df[group_by].unique():
        group_data = df[df[group_by] == group]
        grouped_stats[group] = _calculate_statistics(group_data, columns)
    
    return grouped_stats

def _plot_single_statistic(stats_data, columns, stat_type, color, show_values):
    """Plot a single statistic type."""
    values = stats_data[stat_type]
    
    colors = sns.color_palette(color, n_colors=len(columns))
    bars = plt.bar(columns, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    if show_values:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Columns', fontsize=12)
    ylabel = {'mean': 'Mean', 'std': 'Standard Deviation', 'variance': 'Variance'}
    plt.ylabel(ylabel.get(stat_type, stat_type.title()), fontsize=12)
    plt.xticks(rotation=45, ha='right')

def _plot_single_grouped_statistic(grouped_stats, columns, group_by, stat_type, color, show_values):
    """Plot a single statistic type with grouping."""
    x = np.arange(len(columns))
    width = 0.8 / len(grouped_stats)
    
    colors = sns.color_palette(palette=color, hue=x, n_colors=len(grouped_stats),legend=False)
    
    for i, (group, stats) in enumerate(grouped_stats.items()):
        values = stats[stat_type]
        offset = (i - len(grouped_stats)/2 + 0.5) * width
        bars = plt.bar(x + offset, values, width, label=str(group), 
                      color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
        
        if show_values:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Columns', fontsize=12)
    ylabel = {'mean': 'Mean', 'std': 'Standard Deviation', 'variance': 'Variance'}
    plt.ylabel(ylabel.get(stat_type, stat_type.title()), fontsize=12)
    plt.xticks(x, columns, rotation=45, ha='right')
    plt.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc='upper left')

def _plot_all_statistics(axes, stats_data, columns, color, show_values):
    """Plot all three statistics in subplots."""
    stat_types = ['mean', 'std', 'variance']
    titles = ['Mean', 'Standard Deviation', 'Variance']
    
    colors = sns.color_palette(color, n_colors=len(columns))

    for ax, stat_type, title in zip(axes, stat_types, titles):
        values = stats_data[stat_type]
        bars = ax.bar(columns, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')

def _plot_grouped_statistics(axes, grouped_stats, columns, group_by, color, show_values):
    """Plot all three statistics with grouping in subplots."""
    stat_types = ['mean', 'std', 'variance']
    titles = ['Mean', 'Standard Deviation', 'Variance']
    
    x = np.arange(len(columns))
    width = 0.8 / len(grouped_stats)
    colors = sns.color_palette(palette=color, n_colors=len(grouped_stats), hue=x, legend=False)
    
    for ax, stat_type, title in zip(axes, stat_types, titles):
        for i, (group, stats) in enumerate(grouped_stats.items()):
            values = stats[stat_type]
            offset = (i - len(grouped_stats)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=str(group), 
                         color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
            
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Columns', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation=45, ha='right')
        if ax == axes[0]:
            ax.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

def _print_statistics_table(df, columns, group_by=None):
    """Print a formatted table of statistics."""
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)
    
    if group_by:
        for group in df[group_by].unique():
            print(f"\nGroup: {group}")
            print("-"*70)
            group_data = df[df[group_by] == group]
            _print_column_stats(group_data, columns)
    else:
        _print_column_stats(df, columns)

def _print_column_stats(df, columns):
    """Print statistics for columns."""
    print(f"{'Column':<20} {'Count':<10} {'Mean':<12} {'Std Dev':<12} {'Variance':<12}")
    print("-"*70)
    
    for col in columns:
        count = df[col].count()
        mean = df[col].mean()
        std = df[col].std()
        var = df[col].var()
        print(f"{col:<20} {count:<10} {mean:<12.2f} {std:<12.2f} {var:<12.2f}")

def main():
    """Main function to handle command line arguments and run the script."""
    
    parser = argparse.ArgumentParser(
        description='Compare statistics (mean, std, variance) across multiple CSV columns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Statistic types:
  mean     - Compare means across columns
  std      - Compare standard deviations
  variance - Compare variances
  all      - Show all three statistics in subplots

Examples:
  %(prog)s data.csv column1 column2 column3 --type mean
  %(prog)s data.csv age height weight --type all -o stats.png
  %(prog)s data.csv score1 score2 --type std --group-by class
        """
    )
    
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('columns', nargs='+', help='Column names to compare (space-separated)')
    parser.add_argument('-o', '--output', help='Output file path (e.g., stats_comparison.png)')
    parser.add_argument('-t', '--type', default='mean', 
                       choices=['mean', 'std', 'variance', 'all'],
                       help='Type of statistic to compare (default: mean)')
    parser.add_argument('--width', type=float, default=10, help='Figure width (default: 10)')
    parser.add_argument('--height', type=float, default=6, help='Figure height (default: 6)')
    parser.add_argument('--title', help='Custom title for the plot')
    parser.add_argument('-c', '--color', default='deep', 
                       help='Color palette (default: Set2). Options: deep, muted, bright, pastel, dark, colorblind')
    parser.add_argument('--no-values', action='store_true', 
                       help='Hide values on bars')
    parser.add_argument('--group-by', help='Column name to group comparisons by')
    parser.add_argument('--separator', type=str, default=',',
                       help='Separator used in CSV file (default: ",")')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    if len(args.columns) < 2:
        print("Error: Please provide at least 2 columns to compare.")
        sys.exit(1)
    
    # Create the statistics comparison
    success = create_statistics_comparison(
        csv_file=args.csv_file,
        columns=args.columns,
        separator=args.separator,
        output_file=args.output,
        stat_type=args.type,
        figsize=(args.width, args.height),
        title=args.title,
        color=args.color,
        show_values=not args.no_values,
        group_by=args.group_by
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
