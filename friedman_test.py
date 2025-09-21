import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import sys
import os
from scipy.stats import friedmanchisquare, chi2

def perform_friedman_test(csv_file, columns=None, alpha=0.05, separator=',', 
                         data_format='wide', subject_col=None):
    """
    Perform Friedman's test for repeated measures on data from a CSV file.
    
    Parameters:
    - csv_file: Path to CSV file
    - columns: List of column names for conditions (wide format) or condition column name (long format)
    - alpha: Significance level (default: 0.05)
    - separator: CSV separator (default: ',')
    - data_format: 'wide' (each column is a condition) or 'long' (conditions in one column)
    - subject_col: Subject identifier column (for long format)
    
    Returns:
    - Dictionary containing test results and statistics
    """
    try:
        df = pd.read_csv(csv_file, sep=separator)
        
        if data_format == 'wide':
            # Wide format: each column represents a condition
            if columns is None:
                # Use all numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) < 3:
                    raise ValueError("Need at least 3 conditions for Friedman test")
                columns = numeric_columns
            else:
                # Parse comma-separated column names
                if isinstance(columns, str):
                    columns = [col.strip() for col in columns.split(',')]
                
                # Verify columns exist
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Columns not found: {missing_columns}")
                
                # Check if columns are numeric
                numeric_columns = [col for col in columns if df[col].dtype in ['int64', 'float64']]
                if len(numeric_columns) < len(columns):
                    non_numeric = [col for col in columns if col not in numeric_columns]
                    raise ValueError(f"Non-numeric columns found: {non_numeric}")
                columns = numeric_columns
            
            if len(columns) < 3:
                raise ValueError("Friedman test requires at least 3 conditions")
            
            clean_df = df[columns].dropna()
            
            if len(clean_df) < 3:
                raise ValueError("Need at least 3 complete observations for Friedman test")
            
            # Prepare data for Friedman test
            data_arrays = [clean_df[col].values for col in columns]
            condition_names = columns
            n_subjects = len(clean_df)
            
        else:  # long format
            if columns is None or subject_col is None:
                raise ValueError("For long format, both 'columns' (condition column) and 'subject_col' must be specified")
            
            condition_col = columns
            if condition_col not in df.columns:
                raise ValueError(f"Condition column '{condition_col}' not found")
            if subject_col not in df.columns:
                raise ValueError(f"Subject column '{subject_col}' not found")
            
            # Find the value column (should be numeric)
            value_cols = [col for col in df.columns 
                         if col not in [condition_col, subject_col] and 
                         df[col].dtype in ['int64', 'float64']]
            
            if len(value_cols) != 1:
                raise ValueError("Long format should have exactly one numeric value column")
            
            value_col = value_cols[0]
            
            # Pivot to wide format
            pivot_df = df.pivot(index=subject_col, columns=condition_col, values=value_col)
            
            clean_df = pivot_df.dropna()
            
            if len(clean_df.columns) < 3:
                raise ValueError("Friedman test requires at least 3 conditions")
            
            if len(clean_df) < 3:
                raise ValueError("Need at least 3 complete subjects for Friedman test")
            
            # Prepare data for Friedman test
            condition_names = clean_df.columns.tolist()
            data_arrays = [clean_df[col].values for col in condition_names]
            n_subjects = len(clean_df)
            columns = condition_names
        
        # Perform Friedman's test
        statistic, p_value = friedmanchisquare(*data_arrays)
        
        # Calculate degrees of freedom
        k = len(condition_names)  # number of conditions
        df_friedman = k - 1
        
        # Calculate effect size (Kendall's W)
        # W = χ²/(N(k-1)) where N is number of subjects, k is number of conditions
        kendalls_w = statistic / (n_subjects * (k - 1))
        
        # Calculate ranks for each subject across conditions
        rank_data = np.zeros((n_subjects, k))
        for i in range(n_subjects):
            subject_values = [data_arrays[j][i] for j in range(k)]
            ranks = stats.rankdata(subject_values)
            rank_data[i, :] = ranks
        
        # Calculate mean ranks for each condition
        mean_ranks = np.mean(rank_data, axis=0)
        
        # Calculate sum of ranks for each condition
        sum_ranks = np.sum(rank_data, axis=0)
        
        # Calculate descriptive statistics for each condition
        descriptive_stats = {}
        for i, condition in enumerate(condition_names):
            data = data_arrays[i]
            descriptive_stats[condition] = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'mean_rank': mean_ranks[i],
                'sum_rank': sum_ranks[i]
            }
        
        # Interpret effect size
        if kendalls_w < 0.1:
            effect_size_interpretation = "negligible"
        elif kendalls_w < 0.3:
            effect_size_interpretation = "small"
        elif kendalls_w < 0.5:
            effect_size_interpretation = "medium"
        else:
            effect_size_interpretation = "large"
        
        # Post-hoc analysis preparation (if significant)
        post_hoc_results = None
        if p_value <= alpha and k > 3:
            # Calculate critical difference for post-hoc comparisons
            # Using Nemenyi test approach
            critical_value = stats.chi2.ppf(1 - alpha, df_friedman)
            critical_difference = np.sqrt(k * (k + 1) / (6 * n_subjects)) * np.sqrt(critical_value)
            
            pairwise_comparisons = []
            for i in range(k):
                for j in range(i + 1, k):
                    rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                    significant = rank_diff > critical_difference
                    pairwise_comparisons.append({
                        'condition1': condition_names[i],
                        'condition2': condition_names[j],
                        'rank_diff': rank_diff,
                        'critical_diff': critical_difference,
                        'significant': significant
                    })
            
            post_hoc_results = {
                'critical_difference': critical_difference,
                'pairwise_comparisons': pairwise_comparisons
            }
        
        if p_value <= alpha:
            conclusion = f"There are statistically significant differences between the {k} conditions (χ² = {statistic:.3f}, df = {df_friedman}, p = {p_value:.3f}, W = {kendalls_w:.3f})."
        else:
            conclusion = f"There are no statistically significant differences between the {k} conditions (χ² = {statistic:.3f}, df = {df_friedman}, p = {p_value:.3f}, W = {kendalls_w:.3f})."
        
        results = {
            'test_type': 'Friedman\'s test for repeated measures',
            'statistic': statistic,
            'p_value': p_value,
            'degrees_of_freedom': df_friedman,
            'kendalls_w': kendalls_w,
            'effect_size_interpretation': effect_size_interpretation,
            'n_subjects': n_subjects,
            'n_conditions': k,
            'condition_names': condition_names,
            'mean_ranks': dict(zip(condition_names, mean_ranks)),
            'sum_ranks': dict(zip(condition_names, sum_ranks)),
            'descriptive_stats': descriptive_stats,
            'alpha': alpha,
            'csv_file': csv_file,
            'data_format': data_format,
            'missing_observations': len(df) - n_subjects if data_format == 'wide' else None,
            'post_hoc_results': post_hoc_results,
            'conclusion': conclusion,
            'rank_data': rank_data
        }
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def print_results(results, output_file=None):
    """Print formatted results of Friedman test analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nFriedman's Test for Repeated Measures Results:")
    print("=" * 80)
    print(f"Analysis type: {results['test_type']}")
    print(f"Data file: {results['csv_file']}")
    print(f"Data format: {results['data_format']}")
    print(f"Conditions analyzed: {', '.join(results['condition_names'])}")
    print(f"Significance level (alpha): {results['alpha']}")
    print("=" * 80)

    print("\nTEST DETAILS:")
    print("-" * 40)
    print("Test: Friedman's test for repeated measures")
    print("Null hypothesis: All conditions have identical distributions")
    print("Alternative hypothesis: At least one condition differs from the others")
    print("Advantages: Non-parametric, robust to outliers, handles ordinal data")
    print("Assumptions: Related/matched samples, ordinal or continuous data")
    print("Note: This is the non-parametric alternative to repeated measures ANOVA")
    print("Missing values: Subjects with any missing condition are excluded")

    print(f"\nSAMPLE INFORMATION:")
    print("-" * 40)
    print(f"Number of subjects: {results['n_subjects']}")
    print(f"Number of conditions: {results['n_conditions']}")
    if results['missing_observations'] is not None and results['missing_observations'] > 0:
        print(f"Subjects with missing values removed: {results['missing_observations']}")

    print(f"\nTEST STATISTICS:")
    print("-" * 40)
    print(f"Friedman's χ²: {results['statistic']:.4f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Kendall's W (effect size): {results['kendalls_w']:.4f}")
    print(f"Effect size interpretation: {results['effect_size_interpretation']}")

    print(f"\nRANK STATISTICS:")
    print("-" * 40)
    print(f"{'Condition':<20} {'Mean Rank':<12} {'Sum of Ranks':<15}")
    print("-" * 50)
    for condition in results['condition_names']:
        mean_rank = results['mean_ranks'][condition]
        sum_rank = results['sum_ranks'][condition]
        print(f"{condition:<20} {mean_rank:<12.2f} {sum_rank:<15.0f}")

    print(f"\nDESCRIPTIVE STATISTICS:")
    print("-" * 40)
    for condition in results['condition_names']:
        stats = results['descriptive_stats'][condition]
        print(f"\n{condition}:")
        print(f"  Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Q1: {stats['q25']:.4f}, Q3: {stats['q75']:.4f}")

    if results['post_hoc_results'] and results['p_value'] <= results['alpha']:
        print(f"\nPOST-HOC ANALYSIS (Nemenyi test):")
        print("-" * 40)
        print(f"Critical difference: {results['post_hoc_results']['critical_difference']:.4f}")
        print(f"\nPairwise comparisons:")
        print(f"{'Condition 1':<15} {'Condition 2':<15} {'Rank Diff':<12} {'Significant':<12}")
        print("-" * 60)
        
        significant_pairs = []
        for comparison in results['post_hoc_results']['pairwise_comparisons']:
            sig_text = "Yes" if comparison['significant'] else "No"
            print(f"{comparison['condition1']:<15} {comparison['condition2']:<15} "
                  f"{comparison['rank_diff']:<12.4f} {sig_text:<12}")
            if comparison['significant']:
                significant_pairs.append(comparison)
        
        if significant_pairs:
            print(f"\nSignificant pairwise differences found: {len(significant_pairs)}")
        else:
            print(f"\nNo significant pairwise differences found.")

    print("\nSTATISTICAL CONCLUSION:")
    print("-" * 40)
    print(f"Conclusion: {results['conclusion']}")

    print("\nINTERPRETATION:")
    print("-" * 40)
    
    if results['p_value'] <= results['alpha']:
        print("The Friedman test indicates significant differences between conditions.")
        
        # Find the condition with highest and lowest mean ranks
        ranks = results['mean_ranks']
        highest_condition = max(ranks.keys(), key=lambda k: ranks[k])
        lowest_condition = min(ranks.keys(), key=lambda k: ranks[k])
        
        print(f"Highest mean rank: {highest_condition} ({ranks[highest_condition]:.2f})")
        print(f"Lowest mean rank: {lowest_condition} ({ranks[lowest_condition]:.2f})")
        
        # Effect size interpretation
        w = results['kendalls_w']
        print(f"\nKendall's W = {w:.3f} indicates {results['effect_size_interpretation']} agreement among subjects.")
        print(f"This means approximately {w*100:.1f}% of the variance in ranks is due to condition differences.")
        
        if results['p_value'] <= 0.001:
            print("The result is highly significant (p ≤ 0.001).")
        elif results['p_value'] <= 0.01:
            print("The result is very significant (p ≤ 0.01).")
        else:
            print("The result is significant (p ≤ 0.05).")
            
        if results['post_hoc_results']:
            significant_pairs = [c for c in results['post_hoc_results']['pairwise_comparisons'] if c['significant']]
            if significant_pairs:
                print(f"\nPost-hoc analysis revealed {len(significant_pairs)} significant pairwise differences.")
                print("These pairs show statistically significant rank differences:")
                for pair in significant_pairs:
                    print(f"  - {pair['condition1']} vs {pair['condition2']}")
            else:
                print("\nPost-hoc analysis found no significant pairwise differences.")
                print("The overall significant result may be due to small differences across multiple pairs.")
        else:
            print("\nWith 3 conditions, no post-hoc testing is needed - all pairs can be considered different.")
    
    else:
        print("The Friedman test found no significant differences between conditions.")
        print("This could mean:")
        print("- The conditions truly have similar effects")
        print("- The differences are too small to detect with this sample size")
        print("- The variability between subjects is masking condition effects")
        print("- The data violates the test assumptions")
        
        w = results['kendalls_w']
        print(f"\nKendall's W = {w:.3f} indicates {results['effect_size_interpretation']} agreement,")
        print(f"suggesting {w*100:.1f}% of rank variance is due to systematic condition differences.")

    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")
        if sys.platform == "win32":
            os.startfile(output_file)  # Open the output file automatically on Windows

def main():
    parser = argparse.ArgumentParser(description='Perform Friedman\'s test for repeated measures on data from CSV files.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('columns', nargs='?', 
                        help='Comma-separated column names for conditions (wide format) or condition column name (long format)')
    parser.add_argument('--data_format', choices=['wide', 'long'], 
                        default='wide', help='Data format: wide (each column is a condition) or long (conditions in one column)')
    parser.add_argument('--subject_col', type=str,
                        help='Subject identifier column name (required for long format)')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='Significance level (default: 0.05)')
    parser.add_argument('--separator', type=str, default=',',
                        help='CSV separator (default: ",")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    if args.data_format == 'long' and args.subject_col is None:
        parser.error("--subject_col is required for long format data")

    results = perform_friedman_test(
        args.csv_file, args.columns, args.alpha, 
        args.separator, args.data_format, args.subject_col
    )
    
    print_results(results, args.output)

if __name__ == "__main__":
    main()