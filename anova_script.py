import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, levene, shapiro
import argparse
import sys
import os
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def check_assumptions(data_arrays, group_names, alpha=0.05):
    """
    Check ANOVA assumptions: normality and homogeneity of variance.
    
    Parameters:
    - data_arrays: List of arrays, one for each group
    - group_names: List of group names
    - alpha: Significance level for assumption tests
    
    Returns:
    - Dictionary with assumption test results
    """
    assumptions = {
        'normality_test': {},
        'homogeneity_test': {},
        'assumptions_met': True,
        'warnings': []
    }
    
    # Test normality for each group (Shapiro-Wilk)
    normality_violations = []
    for i, (data, name) in enumerate(zip(data_arrays, group_names)):
        if len(data) >= 3:  # Shapiro-Wilk requires at least 3 observations
            stat, p_val = shapiro(data)
            assumptions['normality_test'][name] = {
                'statistic': stat,
                'p_value': p_val,
                'normal': p_val > alpha
            }
            if p_val <= alpha:
                normality_violations.append(name)
        else:
            assumptions['normality_test'][name] = {
                'statistic': None,
                'p_value': None,
                'normal': None,
                'note': 'Too few observations for normality test'
            }
    
    # Test homogeneity of variance (Levene's test)
    if len(data_arrays) >= 2 and all(len(arr) >= 2 for arr in data_arrays):
        levene_stat, levene_p = levene(*data_arrays)
        assumptions['homogeneity_test'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'homogeneous': levene_p > alpha
        }
        
        if levene_p <= alpha:
            assumptions['assumptions_met'] = False
            assumptions['warnings'].append("Homogeneity of variance assumption violated (Levene's test significant)")
    
    if normality_violations:
        assumptions['assumptions_met'] = False
        assumptions['warnings'].append(f"Normality assumption violated in groups: {', '.join(normality_violations)}")
    
    return assumptions

def calculate_effect_size(f_stat, df_between, df_within, n_total):
    """Calculate effect sizes for ANOVA."""
    # Eta-squared
    eta_squared = (df_between * f_stat) / (df_between * f_stat + df_within)
    
    # Partial eta-squared (same as eta-squared for one-way ANOVA)
    partial_eta_squared = eta_squared
    
    # Cohen's f
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
    
    # Omega-squared (less biased estimate)
    omega_squared = (df_between * (f_stat - 1)) / (df_between * (f_stat - 1) + n_total)
    omega_squared = max(0, omega_squared)  # Can't be negative
    
    # Interpret effect sizes
    if eta_squared < 0.01:
        eta_interpretation = "negligible"
    elif eta_squared < 0.06:
        eta_interpretation = "small"
    elif eta_squared < 0.14:
        eta_interpretation = "medium"
    else:
        eta_interpretation = "large"
    
    if cohens_f < 0.1:
        cohens_interpretation = "negligible"
    elif cohens_f < 0.25:
        cohens_interpretation = "small"
    elif cohens_f < 0.4:
        cohens_interpretation = "medium"
    else:
        cohens_interpretation = "large"
    
    return {
        'eta_squared': eta_squared,
        'partial_eta_squared': partial_eta_squared,
        'omega_squared': omega_squared,
        'cohens_f': cohens_f,
        'eta_interpretation': eta_interpretation,
        'cohens_interpretation': cohens_interpretation
    }

def tukey_hsd_post_hoc(data_arrays, group_names, alpha=0.05):
    """
    Perform Tukey's HSD post-hoc test for pairwise comparisons.
    """
    from scipy.stats import studentized_range
    
    # Calculate group statistics
    group_means = [np.mean(arr) for arr in data_arrays]
    group_sizes = [len(arr) for arr in data_arrays]
    
    # Calculate MSE (Mean Square Error) from within-group variance
    all_data = np.concatenate(data_arrays)
    grand_mean = np.mean(all_data)
    
    # Calculate within-group sum of squares
    ss_within = 0
    for data in data_arrays:
        ss_within += np.sum((data - np.mean(data))**2)
    
    df_within = sum(group_sizes) - len(data_arrays)
    mse = ss_within / df_within if df_within > 0 else 0
    
    # Perform pairwise comparisons
    comparisons = []
    k = len(data_arrays)  # number of groups
    
    # Critical value from studentized range distribution
    try:
        q_critical = studentized_range.ppf(1 - alpha, k, df_within)
    except:
        # Fallback approximation if studentized_range not available
        q_critical = 3.0  # Conservative approximation
    
    for i, j in combinations(range(k), 2):
        mean_diff = abs(group_means[i] - group_means[j])
        
        # Standard error for the difference
        n_i, n_j = group_sizes[i], group_sizes[j]
        se_diff = np.sqrt(mse * (1/n_i + 1/n_j) / 2)
        
        # Critical difference
        critical_diff = q_critical * se_diff
        
        # Test significance
        significant = mean_diff > critical_diff
        
        # Calculate p-value approximation
        if se_diff > 0:
            q_stat = mean_diff / se_diff
            # This is an approximation - exact p-values require more complex calculations
            p_value = 1 - stats.norm.cdf(abs(q_stat))  # Very rough approximation
        else:
            q_stat = 0
            p_value = 1.0
        
        comparisons.append({
            'group1': group_names[i],
            'group2': group_names[j],
            'mean_diff': mean_diff,
            'se_diff': se_diff,
            'critical_diff': critical_diff,
            'q_statistic': q_stat,
            'p_value': p_value,
            'significant': significant
        })
    
    return {
        'method': 'Tukey HSD',
        'alpha': alpha,
        'comparisons': comparisons,
        'mse': mse,
        'df_within': df_within
    }

def perform_anova_test(csv_file, columns=None, alpha=0.05, separator=',', 
                      data_format='wide', subject_col=None, group_col=None,
                      anova_type='one_way', check_assumptions_flag=True):
    """
    Perform ANOVA test on data from a CSV file.
    
    Parameters:
    - csv_file: Path to CSV file
    - columns: List of column names for groups (wide format) or value column name (long format)
    - alpha: Significance level (default: 0.05)
    - separator: CSV separator (default: ',')
    - data_format: 'wide' (each column is a group) or 'long' (groups in one column)
    - subject_col: Subject identifier column (for repeated measures)
    - group_col: Group identifier column (for long format)
    - anova_type: 'one_way', 'repeated_measures', or 'two_way'
    - check_assumptions_flag: Whether to check ANOVA assumptions
    
    Returns:
    - Dictionary containing test results and statistics
    """
    try:
        df = pd.read_csv(csv_file, sep=separator)
        
        if data_format == 'wide':
            # Wide format: each column represents a group/condition
            if columns is None:
                # Use all numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) < 2:
                    raise ValueError("Need at least 2 groups for ANOVA")
                columns = numeric_columns
            else:
                # Parse comma-separated column names
                if isinstance(columns, str):
                    columns = [col.strip() for col in columns.split(',')]
                
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Columns not found: {missing_columns}")
                
                numeric_columns = [col for col in columns if df[col].dtype in ['int64', 'float64']]
                if len(numeric_columns) < len(columns):
                    non_numeric = [col for col in columns if col not in numeric_columns]
                    raise ValueError(f"Non-numeric columns found: {non_numeric}")
                columns = numeric_columns
            
            if len(columns) < 2:
                raise ValueError("ANOVA requires at least 2 groups")
            
            clean_df = df[columns].dropna()
            
            if len(clean_df) < 6:  # Minimum for meaningful ANOVA
                raise ValueError("Need at least 6 complete observations for ANOVA")
            
            data_arrays = [clean_df[col].values for col in columns]
            group_names = columns
            n_total = len(clean_df)
            
        else:  # long format
            if columns is None or group_col is None:
                raise ValueError("For long format, both 'columns' (value column) and 'group_col' must be specified")
            
            value_col = columns
            if value_col not in df.columns:
                raise ValueError(f"Value column '{value_col}' not found")
            if group_col not in df.columns:
                raise ValueError(f"Group column '{group_col}' not found")
            
            if df[value_col].dtype not in ['int64', 'float64']:
                raise ValueError(f"Value column '{value_col}' must be numeric")
            
            clean_df = df[[value_col, group_col]].dropna()
            
            groups = clean_df[group_col].unique()
            if len(groups) < 2:
                raise ValueError("ANOVA requires at least 2 groups")
            
            data_arrays = []
            group_names = []
            for group in sorted(groups):
                group_data = clean_df[clean_df[group_col] == group][value_col].values
                if len(group_data) > 0:
                    data_arrays.append(group_data)
                    group_names.append(str(group))
            
            n_total = len(clean_df)
            columns = group_names
        
        # Check if we have enough data in each group
        min_group_size = min(len(arr) for arr in data_arrays)
        if min_group_size < 2:
            raise ValueError("Each group must have at least 2 observations")
        
        # Perform one-way ANOVA
        f_statistic, p_value = f_oneway(*data_arrays)
        
        # Calculate degrees of freedom
        k = len(group_names)  # number of groups
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1
        
        # Calculate sum of squares
        all_data = np.concatenate(data_arrays)
        grand_mean = np.mean(all_data)
        
        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)
        
        # Between-group sum of squares
        ss_between = 0
        for data in data_arrays:
            group_mean = np.mean(data)
            ss_between += len(data) * (group_mean - grand_mean)**2
        
        # Within-group sum of squares
        ss_within = ss_total - ss_between
        
        # Mean squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        # Calculate effect sizes
        effect_sizes = calculate_effect_size(f_statistic, df_between, df_within, n_total)
        
        # Calculate descriptive statistics for each group
        descriptive_stats = {}
        for i, group_name in enumerate(group_names):
            data = data_arrays[i]
            descriptive_stats[group_name] = {
                'n': len(data),
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data, ddof=1),
                'se': np.std(data, ddof=1) / np.sqrt(len(data)),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                '95_ci_lower': np.mean(data) - 1.96 * np.std(data, ddof=1) / np.sqrt(len(data)),
                '95_ci_upper': np.mean(data) + 1.96 * np.std(data, ddof=1) / np.sqrt(len(data))
            }
        
        # Check assumptions
        assumption_results = None
        if check_assumptions_flag:
            assumption_results = check_assumptions(data_arrays, group_names, alpha)
        
        # Post-hoc analysis (if significant and more than 2 groups)
        post_hoc_results = None
        if p_value <= alpha and k > 2:
            post_hoc_results = tukey_hsd_post_hoc(data_arrays, group_names, alpha)
        
        # Generate conclusion
        if p_value <= alpha:
            conclusion = f"There are statistically significant differences between the {k} groups (F({df_between}, {df_within}) = {f_statistic:.3f}, p = {p_value:.3f}, η² = {effect_sizes['eta_squared']:.3f})."
        else:
            conclusion = f"There are no statistically significant differences between the {k} groups (F({df_between}, {df_within}) = {f_statistic:.3f}, p = {p_value:.3f}, η² = {effect_sizes['eta_squared']:.3f})."
        
        results = {
            'test_type': 'One-way ANOVA',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'df_total': df_total,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'ss_total': ss_total,
            'ms_between': ms_between,
            'ms_within': ms_within,
            'effect_sizes': effect_sizes,
            'n_total': n_total,
            'n_groups': k,
            'group_names': group_names,
            'descriptive_stats': descriptive_stats,
            'assumption_results': assumption_results,
            'alpha': alpha,
            'csv_file': csv_file,
            'data_format': data_format,
            'missing_observations': len(df) - n_total if data_format == 'wide' else None,
            'post_hoc_results': post_hoc_results,
            'conclusion': conclusion,
            'raw_data': data_arrays
        }
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def print_results(results, output_file=None):
    """Print formatted results of ANOVA analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nOne-Way Analysis of Variance (ANOVA) Results:")
    print("=" * 80)
    print(f"Analysis type: {results['test_type']}")
    print(f"Data file: {results['csv_file']}")
    print(f"Data format: {results['data_format']}")
    print(f"Groups analyzed: {', '.join(results['group_names'])}")
    print(f"Significance level (alpha): {results['alpha']}")
    print("=" * 80)

    print("\nTEST DETAILS:")
    print("-" * 40)
    print("Test: One-way Analysis of Variance (ANOVA)")
    print("Null hypothesis: All group means are equal (μ₁ = μ₂ = ... = μₖ)")
    print("Alternative hypothesis: At least one group mean differs from the others")
    print("Assumptions: 1) Independence of observations")
    print("             2) Normality of residuals")
    print("             3) Homogeneity of variances (homoscedasticity)")
    print("             4) Interval or ratio level dependent variable")

    print(f"\nSAMPLE INFORMATION:")
    print("-" * 40)
    print(f"Total sample size: {results['n_total']}")
    print(f"Number of groups: {results['n_groups']}")
    if results['missing_observations'] is not None and results['missing_observations'] > 0:
        print(f"Observations with missing values removed: {results['missing_observations']}")

    print(f"\nGROUP SIZES:")
    print("-" * 40)
    for group_name in results['group_names']:
        n = results['descriptive_stats'][group_name]['n']
        print(f"{group_name}: {n} observations")

    print(f"\nANOVA TABLE:")
    print("-" * 80)
    print(f"{'Source':<15} {'SS':<12} {'df':<8} {'MS':<12} {'F':<10} {'p-value':<10}")
    print("-" * 80)
    print(f"{'Between Groups':<15} {results['ss_between']:<12.4f} {results['df_between']:<8} "
          f"{results['ms_between']:<12.4f} {results['f_statistic']:<10.4f} {results['p_value']:<10.4f}")
    print(f"{'Within Groups':<15} {results['ss_within']:<12.4f} {results['df_within']:<8} "
          f"{results['ms_within']:<12.4f}")
    print(f"{'Total':<15} {results['ss_total']:<12.4f} {results['df_total']:<8}")

    print(f"\nEFFECT SIZES:")
    print("-" * 40)
    es = results['effect_sizes']
    print(f"Eta-squared (η²): {es['eta_squared']:.4f} ({es['eta_interpretation']} effect)")
    print(f"Omega-squared (ω²): {es['omega_squared']:.4f} (less biased estimate)")
    print(f"Cohen's f: {es['cohens_f']:.4f} ({es['cohens_interpretation']} effect)")
    print(f"")
    print(f"Effect size interpretation (η²): negligible < 0.01, small < 0.06, medium < 0.14, large ≥ 0.14")
    print(f"Effect size interpretation (f): negligible < 0.1, small < 0.25, medium < 0.4, large ≥ 0.4")

    print(f"\nDESCRIPTIVE STATISTICS:")
    print("-" * 40)
    print(f"{'Group':<15} {'N':<5} {'Mean':<8} {'SD':<8} {'SE':<8} {'95% CI':<20}")
    print("-" * 70)
    for group_name in results['group_names']:
        stats = results['descriptive_stats'][group_name]
        ci_text = f"[{stats['95_ci_lower']:.2f}, {stats['95_ci_upper']:.2f}]"
        print(f"{group_name:<15} {stats['n']:<5} {stats['mean']:<8.3f} {stats['std']:<8.3f} "
              f"{stats['se']:<8.3f} {ci_text:<20}")

    print(f"\nDETAILED DESCRIPTIVE STATISTICS:")
    print("-" * 40)
    for group_name in results['group_names']:
        stats = results['descriptive_stats'][group_name]
        print(f"\n{group_name}:")
        print(f"  N: {stats['n']}, Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}, Std Error: {stats['se']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Q1: {stats['q25']:.4f}, Q3: {stats['q75']:.4f}")

    if results['assumption_results']:
        print(f"\nASSUMPTION CHECKS:")
        print("-" * 40)
        
        assumptions = results['assumption_results']
        
        print("Normality Tests (Shapiro-Wilk):")
        print(f"{'Group':<15} {'W-statistic':<12} {'p-value':<10} {'Normal':<8}")
        print("-" * 50)
        for group_name in results['group_names']:
            norm_test = assumptions['normality_test'][group_name]
            if norm_test['statistic'] is not None:
                normal_text = "Yes" if norm_test['normal'] else "No"
                print(f"{group_name:<15} {norm_test['statistic']:<12.4f} "
                      f"{norm_test['p_value']:<10.4f} {normal_text:<8}")
            else:
                print(f"{group_name:<15} {'N/A':<12} {'N/A':<10} {'N/A':<8}")
        
        if 'homogeneity_test' in assumptions and assumptions['homogeneity_test']:
            homo_test = assumptions['homogeneity_test']
            print(f"\nHomogeneity of Variance Test (Levene's):")
            print(f"Levene's statistic: {homo_test['statistic']:.4f}")
            print(f"p-value: {homo_test['p_value']:.4f}")
            homo_text = "Yes" if homo_test['homogeneous'] else "No"
            print(f"Variances equal: {homo_text}")
        
        if assumptions['warnings']:
            print(f"\nASSUMPTION WARNINGS:")
            for warning in assumptions['warnings']:
                print(f"⚠ {warning}")
            
            if not assumptions['assumptions_met']:
                print(f"\nRECOMMENDATIONS:")
                print("- Consider non-parametric alternatives (e.g., Kruskal-Wallis test)")
                print("- Check for outliers and consider their removal or transformation")
                print("- Consider data transformations (log, square root, etc.)")
                print("- Use Welch's ANOVA if variances are unequal")

    if results['post_hoc_results'] and results['p_value'] <= results['alpha']:
        print(f"\nPOST-HOC ANALYSIS:")
        print("-" * 40)
        post_hoc = results['post_hoc_results']
        print(f"Method: {post_hoc['method']}")
        print(f"Family-wise alpha: {post_hoc['alpha']}")
        print(f"MSE: {post_hoc['mse']:.4f}")
        
        print(f"\nPairwise Comparisons:")
        print(f"{'Group 1':<15} {'Group 2':<15} {'Mean Diff':<10} {'Significant':<12}")
        print("-" * 60)
        
        significant_pairs = []
        for comparison in post_hoc['comparisons']:
            sig_text = "Yes" if comparison['significant'] else "No"
            print(f"{comparison['group1']:<15} {comparison['group2']:<15} "
                  f"{comparison['mean_diff']:<10.4f} {sig_text:<12}")
            if comparison['significant']:
                significant_pairs.append(comparison)
        
        if significant_pairs:
            print(f"\nSignificant pairwise differences found: {len(significant_pairs)}")
            for pair in significant_pairs:
                print(f"  - {pair['group1']} vs {pair['group2']} (difference: {pair['mean_diff']:.3f})")
        else:
            print(f"\nNo significant pairwise differences found.")

    print("\nSTATISTICAL CONCLUSION:")
    print("-" * 40)
    print(f"Conclusion: {results['conclusion']}")

    print("\nINTERPRETATION:")
    print("-" * 40)
    
    if results['p_value'] <= results['alpha']:
        print("The ANOVA test indicates significant differences between group means.")
        
        means = {name: results['descriptive_stats'][name]['mean'] for name in results['group_names']}
        highest_group = max(means.keys(), key=lambda k: means[k])
        lowest_group = min(means.keys(), key=lambda k: means[k])
        
        print(f"Highest mean: {highest_group} ({means[highest_group]:.3f})")
        print(f"Lowest mean: {lowest_group} ({means[lowest_group]:.3f})")
        
        eta_sq = results['effect_sizes']['eta_squared']
        print(f"\nEta-squared = {eta_sq:.3f} indicates {results['effect_sizes']['eta_interpretation']} effect size.")
        print(f"This means approximately {eta_sq*100:.1f}% of the variance is explained by group membership.")
        
        if results['p_value'] <= 0.001:
            print("The result is highly significant (p ≤ 0.001).")
        elif results['p_value'] <= 0.01:
            print("The result is very significant (p ≤ 0.01).")
        else:
            print("The result is significant (p ≤ 0.05).")
            
        if results['post_hoc_results']:
            significant_pairs = [c for c in results['post_hoc_results']['comparisons'] if c['significant']]
            if significant_pairs:
                print(f"\nPost-hoc analysis revealed {len(significant_pairs)} significant pairwise differences.")
            else:
                print("\nPost-hoc analysis found no significant pairwise differences.")
                print("The overall significant result may be due to small differences across multiple pairs.")
        elif results['n_groups'] == 2:
            print("\nWith only 2 groups, the significant ANOVA indicates these groups differ significantly.")
    
    else:
        print("The ANOVA test found no significant differences between group means.")
        print("This could mean:")
        print("- The groups truly have similar means")
        print("- The differences are too small to detect with this sample size")
        print("- The within-group variability is masking between-group differences")
        print("- The data violates ANOVA assumptions")
        
        eta_sq = results['effect_sizes']['eta_squared']
        print(f"\nEta-squared = {eta_sq:.3f} indicates {results['effect_sizes']['eta_interpretation']} effect,")
        print(f"suggesting {eta_sq*100:.1f}% of variance is explained by group differences.")

    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")
        if sys.platform == "win32":
            os.startfile(output_file)  # Open the output file automatically on Windows

def main():
    parser = argparse.ArgumentParser(description='Perform ANOVA test on data from CSV files.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('columns', nargs='?', 
                        help='Comma-separated column names for groups (wide format) or value column name (long format)')
    parser.add_argument('--data_format', choices=['wide', 'long'], 
                        default='wide', help='Data format: wide (each column is a group) or long (groups in one column)')
    parser.add_argument('--group_col', type=str,
                        help='Group identifier column name (required for long format)')
    parser.add_argument('--subject_col', type=str,
                        help='Subject identifier column name (for repeated measures, future feature)')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='Significance level (default: 0.05)')
    parser.add_argument('--separator', type=str, default=',',
                        help='CSV separator (default: ",")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    parser.add_argument('--no_assumptions', action='store_true',
                        help='Skip assumption checking (faster, less thorough)')
    
    args = parser.parse_args()
    
    if args.data_format == 'long' and args.group_col is None:
        parser.error("--group_col is required for long format data")
    
    results = perform_anova_test(
        args.csv_file, args.columns, args.alpha, 
        args.separator, args.data_format, args.subject_col, 
        args.group_col, 'one_way', not args.no_assumptions
    )
    
    print_results(results, args.output)

if __name__ == "__main__":
    main()