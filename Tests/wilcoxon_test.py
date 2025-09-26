import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import sys
import os
from scipy.stats import wilcoxon

def perform_wilcoxon_test(csv_file, column1, column2, alpha=0.05, 
                         alternative='two-sided', separator=',', zero_method='wilcox'):
    """
    Perform Wilcoxon signed-rank test on paired data from a CSV file.
    
    Parameters:
    - csv_file: Path to CSV file
    - column1: Name of the first column (before condition)
    - column2: Name of the second column (after condition)
    - alpha: Significance level (default: 0.05)
    - alternative: Type of test ('two-sided', 'less', 'greater')
    - separator: CSV separator (default: ',')
    - zero_method: How to handle zero differences ('wilcox', 'pratt', 'zsplit')
    
    Returns:
    - Dictionary containing test results and statistics
    """
    try:
        df = pd.read_csv(csv_file, sep=separator)
        
        if column1 not in df.columns:
            raise ValueError(f"Column '{column1}' not found in {csv_file}")
        if column2 not in df.columns:
            raise ValueError(f"Column '{column2}' not found in {csv_file}")
        
        data1 = df[column1]
        data2 = df[column2]
        
        combined_df = pd.DataFrame({column1: data1, column2: data2})
        clean_df = combined_df.dropna()
        
        if len(clean_df) < 6:
            raise ValueError("Need at least 6 pairs of observations for Wilcoxon signed-rank test")
        
        clean_data1 = clean_df[column1]
        clean_data2 = clean_df[column2]
        
        differences = clean_data2 - clean_data1
        
        zero_differences = np.sum(differences == 0)
        non_zero_differences = differences[differences != 0]
        
        # Perform Wilcoxon signed-rank test
        if len(non_zero_differences) < 6:
            raise ValueError("Need at least 6 non-zero differences for reliable test results")
        
        # Use scipy's wilcoxon test
        statistic, p_value = wilcoxon(clean_data1, clean_data2, 
                                    zero_method=zero_method, 
                                    alternative=alternative)
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(clean_df)
        if n > 20:  # Normal approximation is used for large samples
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
            if alternative == 'two-sided' and p_value < alpha:
                # Determine sign based on median difference
                median_diff = np.median(differences)
                z_score = z_score if median_diff > 0 else -z_score
            elif alternative == 'greater' and p_value < alpha:
                z_score = abs(z_score)
            elif alternative == 'less' and p_value < alpha:
                z_score = -abs(z_score)
            
            effect_size = abs(z_score) / np.sqrt(n)
        else:
            effect_size = None
        
        # Calculate descriptive statistics for both conditions
        desc_stats1 = {
            'mean': clean_data1.mean(),
            'median': clean_data1.median(),
            'std': clean_data1.std(),
            'min': clean_data1.min(),
            'max': clean_data1.max(),
            'q25': clean_data1.quantile(0.25),
            'q75': clean_data1.quantile(0.75)
        }
        
        desc_stats2 = {
            'mean': clean_data2.mean(),
            'median': clean_data2.median(),
            'std': clean_data2.std(),
            'min': clean_data2.min(),
            'max': clean_data2.max(),
            'q25': clean_data2.quantile(0.25),
            'q75': clean_data2.quantile(0.75)
        }
        
        # Difference statistics
        diff_stats = {
            'mean': differences.mean(),
            'median': np.median(differences),
            'std': differences.std(),
            'min': differences.min(),
            'max': differences.max(),
            'q25': np.percentile(differences, 25),
            'q75': np.percentile(differences, 75),
            'positive_ranks': np.sum(differences > 0),
            'negative_ranks': np.sum(differences < 0),
            'zero_differences': zero_differences
        }
        
        # Calculate confidence interval for median difference (approximate)
        if n >= 6:
            # Wilcoxon signed-rank confidence interval for median
            sorted_diffs = np.sort(differences)
            # Critical value from Wilcoxon table (approximate)
            if alternative == 'two-sided':
                alpha_adj = alpha / 2
            else:
                alpha_adj = alpha
            
            # For large samples, use normal approximation
            if n > 20:
                z_alpha = stats.norm.ppf(1 - alpha_adj)
                k = int(n/2 - z_alpha * np.sqrt(n/4))
                k = max(0, min(k, n-1))
                
                if k < n and k >= 0:
                    ci_lower = sorted_diffs[k] if k < len(sorted_diffs) else None
                    ci_upper = sorted_diffs[n-1-k] if (n-1-k) < len(sorted_diffs) else None
                else:
                    ci_lower, ci_upper = None, None
            else:
                ci_lower, ci_upper = None, None
        else:
            ci_lower, ci_upper = None, None
        
        results = {
            'test_type': 'Wilcoxon signed-rank test',
            'statistic': statistic,
            'p_value': p_value,
            'sample_size': n,
            'missing_pairs': len(df) - n,
            'alternative': alternative,
            'zero_method': zero_method,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper) if ci_lower is not None else None,
            'descriptive_stats1': desc_stats1,
            'descriptive_stats2': desc_stats2,
            'difference_stats': diff_stats,
            'alpha': alpha,
            'csv_file': csv_file,
            'column1': column1,
            'column2': column2,
            'data1': clean_data1,
            'data2': clean_data2,
            'differences': differences
        }
        
        # Add conclusion
        if p_value <= alpha:
            median_diff = diff_stats['median']
            if alternative == 'two-sided':
                direction = "increase" if median_diff > 0 else "decrease"
                conclusion = f"There is a statistically significant difference between {column1} and {column2}. The median change is a {direction} of {abs(median_diff):.3f} (W = {statistic}, p = {p_value:.3f})."
            elif alternative == 'greater':
                conclusion = f"There is a statistically significant increase from {column1} to {column2}. The median increase is {median_diff:.3f} (W = {statistic}, p = {p_value:.3f})."
            else:  # less
                conclusion = f"There is a statistically significant decrease from {column1} to {column2}. The median decrease is {abs(median_diff):.3f} (W = {statistic}, p = {p_value:.3f})."
        else:
            if alternative == 'two-sided':
                conclusion = f"There is no statistically significant difference between {column1} and {column2} (W = {statistic}, p = {p_value:.3f})."
            elif alternative == 'greater':
                conclusion = f"There is no statistically significant increase from {column1} to {column2} (W = {statistic}, p = {p_value:.3f})."
            else:  # less
                conclusion = f"There is no statistically significant decrease from {column1} to {column2} (W = {statistic}, p = {p_value:.3f})."
        
        results['conclusion'] = conclusion
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def interpret_effect_size(effect_size):
    """Interpret the effect size (r) for Wilcoxon test."""
    if effect_size is None:
        return "Cannot calculate"
    elif effect_size < 0.1:
        return "negligible"
    elif effect_size < 0.3:
        return "small"
    elif effect_size < 0.5:
        return "medium"
    else:
        return "large"

def print_results(results, output_file=None):
    """Print formatted results of Wilcoxon signed-rank test analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nWilcoxon Signed-Rank Test Analysis Results:")
    print("=" * 80)
    print(f"Test type: {results['test_type']}")
    print(f"Data file: {results['csv_file']}")
    print(f"Before condition: {results['column1']}")
    print(f"After condition: {results['column2']}")
    print(f"Alternative hypothesis: {results['alternative']}")
    print(f"Significance level (alpha): {results['alpha']}")
    print(f"Zero differences method: {results['zero_method']}")
    print("=" * 80)

    print("\nTEST DETAILS:")
    print("-" * 40)
    print("Test: Wilcoxon signed-rank test for paired samples")
    
    if results['alternative'] == 'two-sided':
        print("Null hypothesis: Median difference = 0")
        print("Alternative hypothesis: Median difference ≠ 0")
    elif results['alternative'] == 'greater':
        print("Null hypothesis: Median difference ≤ 0")
        print("Alternative hypothesis: Median difference > 0")
    else:  # less
        print("Null hypothesis: Median difference ≥ 0")
        print("Alternative hypothesis: Median difference < 0")
    
    print("Advantages: Non-parametric, robust to outliers, uses rank information")
    print("Assumptions: Paired observations, ordinal or continuous data, symmetric distribution of differences")
    print("Missing values: Pairs with missing values automatically excluded")
    
    print("\nTEST STATISTICS:")
    print("-" * 40)
    print(f"Test statistic (W): {results['statistic']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Sample size (n): {results['sample_size']}")
    if results['missing_pairs'] > 0:
        print(f"Missing pairs removed: {results['missing_pairs']}")
    
    if results['effect_size'] is not None:
        effect_interpretation = interpret_effect_size(results['effect_size'])
        print(f"Effect size (r): {results['effect_size']:.4f} ({effect_interpretation})")
    
    if results['confidence_interval']:
        ci_lower, ci_upper = results['confidence_interval']
        confidence_level = int((1 - results['alpha']) * 100)
        print(f"{confidence_level}% CI for median difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nDESCRIPTIVE STATISTICS:")
    print("-" * 40)
    print(f"{results['column1']} (Before):")
    stats1 = results['descriptive_stats1']
    print(f"  Mean: {stats1['mean']:.4f}, Median: {stats1['median']:.4f}")
    print(f"  Std Dev: {stats1['std']:.4f}")
    print(f"  Range: [{stats1['min']:.4f}, {stats1['max']:.4f}]")
    print(f"  Q1: {stats1['q25']:.4f}, Q3: {stats1['q75']:.4f}")
    
    print(f"\n{results['column2']} (After):")
    stats2 = results['descriptive_stats2']
    print(f"  Mean: {stats2['mean']:.4f}, Median: {stats2['median']:.4f}")
    print(f"  Std Dev: {stats2['std']:.4f}")
    print(f"  Range: [{stats2['min']:.4f}, {stats2['max']:.4f}]")
    print(f"  Q1: {stats2['q25']:.4f}, Q3: {stats2['q75']:.4f}")
    
    print(f"\nDIFFERENCE STATISTICS ({results['column2']} - {results['column1']}):")
    print("-" * 60)
    diff_stats = results['difference_stats']
    print(f"Mean difference: {diff_stats['mean']:.4f}")
    print(f"Median difference: {diff_stats['median']:.4f}")
    print(f"Std Dev of differences: {diff_stats['std']:.4f}")
    print(f"Range: [{diff_stats['min']:.4f}, {diff_stats['max']:.4f}]")
    print(f"Q1: {diff_stats['q25']:.4f}, Q3: {diff_stats['q75']:.4f}")
    print(f"Positive differences: {diff_stats['positive_ranks']} ({diff_stats['positive_ranks']/results['sample_size']*100:.1f}%)")
    print(f"Negative differences: {diff_stats['negative_ranks']} ({diff_stats['negative_ranks']/results['sample_size']*100:.1f}%)")
    print(f"Zero differences: {diff_stats['zero_differences']} ({diff_stats['zero_differences']/results['sample_size']*100:.1f}%)")
    
    print("\nSTATISTICAL CONCLUSION:")
    print("-" * 40)
    print(f"Conclusion: {results['conclusion']}")
    
    print("\nINTERPRETATION:")
    print("-" * 40)
    
    p_val = results['p_value']
    median_diff = diff_stats['median']
    
    if p_val <= results['alpha']:
        if p_val <= 0.001:
            significance_level = "highly significant (p ≤ 0.001)"
        elif p_val <= 0.01:
            significance_level = "very significant (p ≤ 0.01)"
        else:
            significance_level = "significant (p ≤ 0.05)"
        
        print(f"The test result is {significance_level}.")
        
        if results['alternative'] == 'two-sided':
            if median_diff > 0:
                print(f"The median value increased from {results['column1']} to {results['column2']}.")
                print(f"The typical change is an increase of {median_diff:.3f} units.")
            else:
                print(f"The median value decreased from {results['column1']} to {results['column2']}.")
                print(f"The typical change is a decrease of {abs(median_diff):.3f} units.")
        elif results['alternative'] == 'greater':
            print(f"The data supports that {results['column2']} values are significantly greater than {results['column1']} values.")
            print(f"The median increase is {median_diff:.3f} units.")
        else:  # less
            print(f"The data supports that {results['column2']} values are significantly less than {results['column1']} values.")
            print(f"The median decrease is {abs(median_diff):.3f} units.")
        
        if results['effect_size'] is not None:
            effect_interpretation = interpret_effect_size(results['effect_size'])
            print(f"The effect size is {effect_interpretation} (r = {results['effect_size']:.3f}).")
            
            if effect_interpretation == "small":
                print("This suggests a small practical difference.")
            elif effect_interpretation == "medium":
                print("This suggests a moderate practical difference.")
            elif effect_interpretation == "large":
                print("This suggests a large practical difference.")
        
        relative_change = abs(median_diff) / abs(results['descriptive_stats1']['median']) * 100 if results['descriptive_stats1']['median'] != 0 else 0
        print(f"The median relative change is {relative_change:.1f}%.")
        
    else:
        print("No significant difference was detected.")
        print("This could mean:")
        print("- The treatment/intervention had no effect")
        print("- The effect size is too small to detect with this sample size")
        print("- The assumptions of the test are violated")
        print("- There is high variability in the data")
    
    print("\nNOTES:")
    print("-" * 40)
    if diff_stats['zero_differences'] > 0:
        zero_pct = diff_stats['zero_differences'] / results['sample_size'] * 100
        print(f"Zero differences: {zero_pct:.1f}% of pairs showed no change.")
        print(f"Zero differences were handled using the '{results['zero_method']}' method.")
    
    if results['sample_size'] < 20:
        print("Small sample size: Results may be less reliable. Consider collecting more data.")
    
    if results['sample_size'] > 50:
        print("Large sample size: Normal approximation was used for p-value calculation.")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")
        try:
            os.startfile(output_file)  # Open the output file automatically on Windows
        except:
            pass  # Ignore if not on Windows or if startfile is not available

def main():
    parser = argparse.ArgumentParser(description='Perform Wilcoxon signed-rank test on paired data from CSV files.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('column1', help='Name of the first column (before condition)')
    parser.add_argument('column2', help='Name of the second column (after condition)')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='Significance level (default: 0.05)')
    parser.add_argument('--alternative', choices=['two-sided', 'less', 'greater'], 
                        default='two-sided', help='Alternative hypothesis (default: two-sided)')
    parser.add_argument('--zero_method', choices=['wilcox', 'pratt', 'zsplit'],
                        default='wilcox', help='Method for handling zero differences (default: wilcox)')
    parser.add_argument('--separator', type=str, default=',',
                        help='CSV separator (default: ",")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    results = perform_wilcoxon_test(
        args.csv_file, args.column1, args.column2, 
        args.alpha, args.alternative, args.separator, args.zero_method
    )
    
    # Display results
    print_results(results, args.output)

if __name__ == "__main__":
    main()