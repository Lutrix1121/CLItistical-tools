import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import sys
import os
from scipy.stats import spearmanr

def perform_spearman_correlation(csv_file, column1, column2=None, alpha=0.05, 
                                correlation_type='pairwise', separator=','):
    """
    Perform Spearman's rank correlation analysis on data from a CSV file.
    
    Parameters:
    - csv_file: Path to CSV file
    - column1: Name of the first column (or list of columns for matrix)
    - column2: Name of the second column (for pairwise correlation)
    - alpha: Significance level (default: 0.05)
    - correlation_type: Type of analysis ('pairwise' or 'matrix')
    - separator: CSV separator (default: ',')
    
    Returns:
    - Dictionary containing correlation results and statistics
    """
    try:
        df = pd.read_csv(csv_file, sep=separator)
        
        if correlation_type == 'pairwise':
            # Verify that the columns exist
            if column1 not in df.columns:
                raise ValueError(f"Column '{column1}' not found in {csv_file}")
            if column2 is None:
                raise ValueError("Column2 is required for pairwise correlation")
            if column2 not in df.columns:
                raise ValueError(f"Column '{column2}' not found in {csv_file}")
            
            data1 = df[column1]
            data2 = df[column2]
            
            combined_df = pd.DataFrame({column1: data1, column2: data2})
            clean_df = combined_df.dropna()
            
            if len(clean_df) < 3:
                raise ValueError("Need at least 3 pairs of observations for correlation analysis")
            
            clean_data1 = clean_df[column1]
            clean_data2 = clean_df[column2]
            
            # Perform Spearman's correlation
            correlation_coef, p_value = spearmanr(clean_data1, clean_data2)
            
            # Calculate confidence interval (approximation for large samples)
            n = len(clean_data1)
            if n > 10:
                # Fisher z-transformation for confidence interval
                z_score = 0.5 * np.log((1 + correlation_coef) / (1 - correlation_coef))
                se = 1 / np.sqrt(n - 3)
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_lower = z_score - z_alpha * se
                z_upper = z_score + z_alpha * se
                
                # Transform back to correlation scale
                ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            else:
                ci_lower, ci_upper = None, None
            
            # Calculate descriptive statistics
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
            
            results = {
                'correlation_type': 'Pairwise Spearman\'s rank correlation',
                'correlation_coefficient': correlation_coef,
                'p_value': p_value,
                'sample_size': n,
                'missing_pairs': len(df) - n,
                'confidence_interval': (ci_lower, ci_upper) if ci_lower is not None else None,
                'descriptive_stats1': desc_stats1,
                'descriptive_stats2': desc_stats2,
                'alpha': alpha,
                'csv_file': csv_file,
                'column1': column1,
                'column2': column2,
                'data1': clean_data1,
                'data2': clean_data2
            }
            
        else:  # correlation_matrix
            # Get all numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                raise ValueError("Need at least 2 numeric columns for correlation matrix")
            
            # If specific columns provided, use those
            if column1:
                requested_columns = [col.strip() for col in column1.split(',')]
                missing_columns = [col for col in requested_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Columns not found: {missing_columns}")
                numeric_columns = [col for col in requested_columns if col in numeric_columns]
                if len(numeric_columns) < 2:
                    raise ValueError("Need at least 2 numeric columns from specified columns")
            
            clean_df = df[numeric_columns].dropna()
            
            if len(clean_df) < 3:
                raise ValueError("Need at least 3 complete observations for correlation matrix")
            
            # Calculate correlation matrix and p-values
            n_vars = len(numeric_columns)
            correlation_matrix = np.zeros((n_vars, n_vars))
            p_value_matrix = np.ones((n_vars, n_vars))
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                        p_value_matrix[i, j] = 0.0
                    elif i < j:
                        corr, p_val = spearmanr(clean_df.iloc[:, i], clean_df.iloc[:, j])
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
                        p_value_matrix[i, j] = p_val
                        p_value_matrix[j, i] = p_val
            
            # Convert to DataFrames
            corr_df = pd.DataFrame(correlation_matrix, 
                                 index=numeric_columns, 
                                 columns=numeric_columns)
            
            p_val_df = pd.DataFrame(p_value_matrix, 
                                  index=numeric_columns, 
                                  columns=numeric_columns)
            
            # Find significant correlations
            significant_pairs = []
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    if p_value_matrix[i, j] <= alpha:
                        significant_pairs.append({
                            'var1': numeric_columns[i],
                            'var2': numeric_columns[j],
                            'correlation': correlation_matrix[i, j],
                            'p_value': p_value_matrix[i, j]
                        })
            
            results = {
                'correlation_type': 'Spearman\'s rank correlation matrix',
                'correlation_matrix': corr_df,
                'p_value_matrix': p_val_df,
                'sample_size': len(clean_df),
                'missing_observations': len(df) - len(clean_df),
                'variables': numeric_columns,
                'significant_pairs': significant_pairs,
                'alpha': alpha,
                'csv_file': csv_file,
                'column1': column1,
                'column2': None
            }
        
        # Add conclusion
        if correlation_type == 'pairwise':
            if p_value <= alpha:
                strength = interpret_correlation_strength(abs(correlation_coef))
                direction = "positive" if correlation_coef > 0 else "negative"
                conclusion = f"There is a statistically significant {strength} {direction} rank correlation between {column1} and {column2} (ρ = {correlation_coef:.3f}, p = {p_value:.3f})."
            else:
                conclusion = f"There is no statistically significant rank correlation between {column1} and {column2} (ρ = {correlation_coef:.3f}, p = {p_value:.3f})."
        else:
            if significant_pairs:
                conclusion = f"Found {len(significant_pairs)} statistically significant rank correlations out of {len(numeric_columns)*(len(numeric_columns)-1)//2} possible pairs."
            else:
                conclusion = "No statistically significant rank correlations were found among the variables."
        
        results['conclusion'] = conclusion
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def interpret_correlation_strength(abs_corr):
    """Interpret the strength of correlation coefficient."""
    if abs_corr < 0.1:
        return "negligible"
    elif abs_corr < 0.3:
        return "weak"
    elif abs_corr < 0.5:
        return "moderate"
    elif abs_corr < 0.7:
        return "strong"
    else:
        return "very strong"

def print_results(results, output_file=None):
    """Print formatted results of Spearman's correlation analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nSpearman's Rank Correlation Analysis Results:")
    print("=" * 80)
    print(f"Analysis type: {results['correlation_type']}")
    print(f"Data file: {results['csv_file']}")
    if results['correlation_type'].startswith('Pairwise'):
        print(f"Variable 1: {results['column1']}")
        print(f"Variable 2: {results['column2']}")
    else:
        print(f"Variables analyzed: {', '.join(results['variables'])}")
    print(f"Significance level (alpha): {results['alpha']}")
    print("=" * 80)

    print("\nTEST DETAILS:")
    print("-" * 40)
    print("Test: Spearman's rank correlation coefficient")
    print("Null hypothesis: No monotonic relationship exists (ρ = 0)")
    print("Alternative hypothesis: A monotonic relationship exists (ρ ≠ 0)")
    print("Advantages: Non-parametric, robust to outliers, detects monotonic relationships")
    print("Assumptions: Ordinal or continuous data, monotonic relationship")
    print("Missing values: Automatically excluded from analysis")
    
    if results['correlation_type'].startswith('Pairwise'):
        print("\nCORRELATION STATISTICS:")
        print("-" * 40)
        print(f"Spearman's rho (ρ): {results['correlation_coefficient']:.4f}")
        print(f"p-value: {results['p_value']:.4f}")
        print(f"Sample size (n): {results['sample_size']}")
        if results['missing_pairs'] > 0:
            print(f"Missing pairs removed: {results['missing_pairs']}")
        
        if results['confidence_interval']:
            ci_lower, ci_upper = results['confidence_interval']
            print(f"95% Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        print(f"\nDESCRIPTIVE STATISTICS:")
        print("-" * 40)
        print(f"{results['column1']}:")
        stats1 = results['descriptive_stats1']
        print(f"  Mean: {stats1['mean']:.4f}, Median: {stats1['median']:.4f}")
        print(f"  Std Dev: {stats1['std']:.4f}")
        print(f"  Range: [{stats1['min']:.4f}, {stats1['max']:.4f}]")
        print(f"  Q1: {stats1['q25']:.4f}, Q3: {stats1['q75']:.4f}")
        
        print(f"\n{results['column2']}:")
        stats2 = results['descriptive_stats2']
        print(f"  Mean: {stats2['mean']:.4f}, Median: {stats2['median']:.4f}")
        print(f"  Std Dev: {stats2['std']:.4f}")
        print(f"  Range: [{stats2['min']:.4f}, {stats2['max']:.4f}]")
        print(f"  Q1: {stats2['q25']:.4f}, Q3: {stats2['q75']:.4f}")
        
    else:
        print(f"\nSAMPLE INFORMATION:")
        print("-" * 40)
        print(f"Sample size (n): {results['sample_size']}")
        print(f"Number of variables: {len(results['variables'])}")
        if results['missing_observations'] > 0:
            print(f"Observations with missing values removed: {results['missing_observations']}")
        
        print(f"\nCORRELATION MATRIX:")
        print("-" * 40)
        print(results['correlation_matrix'].round(4))
        
        print(f"\nP-VALUE MATRIX:")
        print("-" * 40)
        print(results['p_value_matrix'].round(4))
        
        # Significant correlations
        if results['significant_pairs']:
            print(f"\nSIGNIFICAN CORRELATIONS (p ≤ {results['alpha']}):")
            print("-" * 60)
            print(f"{'Variable 1':<15} {'Variable 2':<15} {'Correlation':<12} {'p-value':<10}")
            print("-" * 60)
            for pair in sorted(results['significant_pairs'], 
                             key=lambda x: abs(x['correlation']), reverse=True):
                print(f"{pair['var1']:<15} {pair['var2']:<15} {pair['correlation']:<12.4f} {pair['p_value']:<10.4f}")
    
    print("\nSTATISTICAL CONCLUSION:")
    print("-" * 40)
    print(f"Conclusion: {results['conclusion']}")
    
    print("\nINTERPRETATION:")
    print("-" * 40)
    
    if results['correlation_type'].startswith('Pairwise'):
        corr_coef = results['correlation_coefficient']
        p_val = results['p_value']
        
        if p_val <= results['alpha']:
            strength = interpret_correlation_strength(abs(corr_coef))
            direction = "positive" if corr_coef > 0 else "negative"
            
            print(f"The correlation is {strength} and {direction}.")
            
            if direction == "positive":
                print(f"As {results['column1']} ranks increase, {results['column2']} ranks tend to increase.")
            else:
                print(f"As {results['column1']} ranks increase, {results['column2']} ranks tend to decrease.")
            
            if p_val <= 0.001:
                print("The correlation is highly significant (p ≤ 0.001).")
            elif p_val <= 0.01:
                print("The correlation is very significant (p ≤ 0.01).")
            else:
                print("The correlation is significant (p ≤ 0.05).")
                
            r_squared = corr_coef ** 2
            print(f"Approximately {r_squared*100:.1f}% of the variance in ranks is shared between the variables.")
            
        else:
            print("No significant monotonic relationship was detected.")
            print("This could mean:")
            print("- The variables are truly independent")
            print("- The relationship is non-monotonic")
            print("- The sample size is too small to detect the relationship")
            print("- The effect size is very small")
    
    else:
        if results['significant_pairs']:
            strongest = max(results['significant_pairs'], key=lambda x: abs(x['correlation']))
            strength = interpret_correlation_strength(abs(strongest['correlation']))
            direction = "positive" if strongest['correlation'] > 0 else "negative"
            
            print(f"Strongest significant correlation: {strongest['var1']} - {strongest['var2']}")
            print(f"This correlation is {strength} and {direction} (ρ = {strongest['correlation']:.3f}).")
            
            # Summary of effect sizes
            strong_pairs = [p for p in results['significant_pairs'] if abs(p['correlation']) >= 0.5]
            moderate_pairs = [p for p in results['significant_pairs'] if 0.3 <= abs(p['correlation']) < 0.5]
            weak_pairs = [p for p in results['significant_pairs'] if abs(p['correlation']) < 0.3]
            
            if strong_pairs:
                print(f"Strong correlations (|ρ| ≥ 0.5): {len(strong_pairs)}")
            if moderate_pairs:
                print(f"Moderate correlations (0.3 ≤ |ρ| < 0.5): {len(moderate_pairs)}")
            if weak_pairs:
                print(f"Weak correlations (|ρ| < 0.3): {len(weak_pairs)}")
        else:
            print("No significant correlations were found among the variables.")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")
        os.startfile(output_file)  # Open the output file automatically on Windows

def main():
    parser = argparse.ArgumentParser(description='Perform Spearman\'s rank correlation analysis on data from CSV files.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('column1', help='Name of the first column (or comma-separated list for matrix analysis)')
    parser.add_argument('--column2', help='Name of the second column (required for pairwise correlation)')
    parser.add_argument('--analysis_type', choices=['pairwise', 'matrix'], 
                        default='pairwise', help='Type of correlation analysis (default: pairwise)')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='Significance level (default: 0.05)')
    parser.add_argument('--separator', type=str, default=',',
                        help='CSV separator (default: ",")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.analysis_type == 'pairwise' and args.column2 is None:
        parser.error("--column2 is required for pairwise correlation analysis")
    
    # Perform Spearman correlation analysis
    results = perform_spearman_correlation(
        args.csv_file, args.column1, args.column2, 
        args.alpha, args.analysis_type, args.separator
    )
    
    # Display results
    print_results(results, args.output)

if __name__ == "__main__":
    main()