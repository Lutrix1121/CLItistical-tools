import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy import optimize
import argparse
import sys
import os
import math
from typing import Dict, Optional, Union

def calculate_cohens_d(mean1, mean2, std1, std2=None, pooled=True):
    """Calculate Cohen's d effect size."""
    if std2 is None:
        std2 = std1
    
    if pooled and std2 is not None:
        pooled_std = math.sqrt(((std1**2) + (std2**2)) / 2)
        return abs(mean1 - mean2) / pooled_std
    else:
        return abs(mean1 - mean2) / std1

def power_analysis_ttest(effect_size=None, sample_size=None, power=None, alpha=0.05, 
                        test_type='two_sample', alternative='two_sided'):
    """
    Perform power analysis for t-tests.
    
    Parameters:
    - effect_size: Cohen's d effect size
    - sample_size: Sample size (per group for two-sample tests)
    - power: Statistical power (1 - β)
    - alpha: Type I error rate
    - test_type: 'one_sample', 'two_sample', or 'paired'
    - alternative: 'two_sided', 'greater', or 'less'
    
    Returns:
    - Dictionary containing power analysis results
    """
    
    # Determine which parameter to solve for
    params = [effect_size, sample_size, power]
    none_count = sum(1 for p in params if p is None)
    
    if none_count != 1:
        raise ValueError("Exactly one of effect_size, sample_size, or power must be None")
    
    # Adjust alpha for one-sided tests
    if alternative in ['greater', 'less']:
        alpha_adj = alpha
    else:
        alpha_adj = alpha / 2
    
    def power_function(d, n):
        """Calculate power given effect size and sample size."""
        if test_type == 'one_sample':
            ncp = d * math.sqrt(n)
            df = n - 1
        elif test_type == 'two_sample':
            ncp = d * math.sqrt(n / 2)
            df = 2 * n - 2
        else:  # paired
            ncp = d * math.sqrt(n)
            df = n - 1
        
        t_crit = stats.t.ppf(1 - alpha_adj, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
        
        if alternative == 'two_sided':
            # For two-sided test, also consider negative tail
            power += stats.nct.cdf(-t_crit, df, ncp)
        
        return power
    
    # Solve for missing parameter
    if effect_size is None:
        # Solve for effect size
        def objective(d):
            return power_function(d, sample_size) - power
        
        try:
            effect_size = optimize.brentq(objective, 0.001, 10.0)
        except ValueError:
            effect_size = float('inf')  # No solution found
    
    elif sample_size is None:
        # Solve for sample size
        def objective(n):
            return power_function(effect_size, int(n)) - power
        
        try:
            sample_size = optimize.brentq(objective, 2, 10000)
            sample_size = math.ceil(sample_size)
        except ValueError:
            sample_size = float('inf')  # No solution found
    
    else:
        # Calculate power
        power = power_function(effect_size, sample_size)
    
    results = {
        'test_type': f'T-test power analysis ({test_type})',
        'effect_size': effect_size,
        'sample_size': sample_size,
        'power': power,
        'alpha': alpha,
        'alternative': alternative,
        'total_sample_size': sample_size * (2 if test_type == 'two_sample' else 1)
    }
    
    return results

def power_analysis_chisquare(effect_size=None, sample_size=None, power=None, alpha=0.05,
                           df=1, test_type='independence'):
    """
    Perform power analysis for chi-square tests.
    
    Parameters:
    - effect_size: Effect size (w for goodness of fit, Cramer's V for independence)
    - sample_size: Total sample size
    - power: Statistical power
    - alpha: Type I error rate
    - df: Degrees of freedom
    - test_type: 'independence' or 'goodness_of_fit'
    """
    
    params = [effect_size, sample_size, power]
    none_count = sum(1 for p in params if p is None)
    
    if none_count != 1:
        raise ValueError("Exactly one of effect_size, sample_size, or power must be None")
    
    def power_function(w, n):
        """Calculate power for chi-square test."""
        ncp = n * w**2  # Non-centrality parameter
        chi2_crit = stats.chi2.ppf(1 - alpha, df)
        power = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)
        return power
    
    if effect_size is None:
        def objective(w):
            return power_function(w, sample_size) - power
        
        try:
            effect_size = optimize.brentq(objective, 0.001, 2.0)
        except ValueError:
            effect_size = float('inf')
    
    elif sample_size is None:
        def objective(n):
            return power_function(effect_size, int(n)) - power
        
        try:
            sample_size = optimize.brentq(objective, 10, 100000)
            sample_size = math.ceil(sample_size)
        except ValueError:
            sample_size = float('inf')
    
    else:
        power = power_function(effect_size, sample_size)
    
    results = {
        'test_type': f'Chi-square power analysis ({test_type})',
        'effect_size': effect_size,
        'sample_size': sample_size,
        'power': power,
        'alpha': alpha,
        'degrees_of_freedom': df
    }
    
    return results

def power_analysis_anova(effect_size=None, sample_size=None, power=None, alpha=0.05,
                        groups=3):
    """
    Perform power analysis for one-way ANOVA.
    
    Parameters:
    - effect_size: Cohen's f effect size
    - sample_size: Sample size per group
    - power: Statistical power
    - alpha: Type I error rate
    - groups: Number of groups
    """
    
    params = [effect_size, sample_size, power]
    none_count = sum(1 for p in params if p is None)
    
    if none_count != 1:
        raise ValueError("Exactly one of effect_size, sample_size, or power must be None")
    
    def power_function(f, n):
        """Calculate power for one-way ANOVA."""
        total_n = n * groups
        ncp = total_n * f**2
        df_between = groups - 1
        df_within = total_n - groups
        
        f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
        power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
        return power
    
    if effect_size is None:
        def objective(f):
            return power_function(f, sample_size) - power
        
        try:
            effect_size = optimize.brentq(objective, 0.001, 2.0)
        except ValueError:
            effect_size = float('inf')
    
    elif sample_size is None:
        def objective(n):
            return power_function(effect_size, int(n)) - power
        
        try:
            sample_size = optimize.brentq(objective, 2, 10000)
            sample_size = math.ceil(sample_size)
        except ValueError:
            sample_size = float('inf')
    
    else:
        power = power_function(effect_size, sample_size)
    
    results = {
        'test_type': 'One-way ANOVA power analysis',
        'effect_size': effect_size,
        'sample_size': sample_size,
        'sample_size_per_group': sample_size,
        'total_sample_size': sample_size * groups,
        'power': power,
        'alpha': alpha,
        'groups': groups
    }
    
    return results

def power_analysis_correlation(effect_size=None, sample_size=None, power=None, alpha=0.05,
                             alternative='two_sided'):
    """
    Perform power analysis for correlation tests.
    
    Parameters:
    - effect_size: Pearson's r correlation coefficient
    - sample_size: Sample size
    - power: Statistical power
    - alpha: Type I error rate
    - alternative: 'two_sided', 'greater', or 'less'
    """
    
    params = [effect_size, sample_size, power]
    none_count = sum(1 for p in params if p is None)
    
    if none_count != 1:
        raise ValueError("Exactly one of effect_size, sample_size, or power must be None")
    
    # Fisher's z transformation
    def fisher_z(r):
        return 0.5 * math.log((1 + r) / (1 - r))
    
    def power_function(r, n):
        """Calculate power for correlation test."""
        if abs(r) >= 1:
            return 1.0 if abs(r) > 0 else alpha
        
        z_r = fisher_z(r)
        se = 1 / math.sqrt(n - 3)
        
        if alternative == 'two_sided':
            z_crit = stats.norm.ppf(1 - alpha / 2)
            power = 2 * (1 - stats.norm.cdf(z_crit - abs(z_r) / se))
        else:
            z_crit = stats.norm.ppf(1 - alpha)
            if alternative == 'greater':
                power = 1 - stats.norm.cdf(z_crit - z_r / se)
            else:  # less
                power = 1 - stats.norm.cdf(z_crit + z_r / se)
        
        return min(power, 1.0)
    
    if effect_size is None:
        def objective(r):
            return power_function(r, sample_size) - power
        
        try:
            effect_size = optimize.brentq(objective, 0.001, 0.999)
        except ValueError:
            effect_size = float('inf')
    
    elif sample_size is None:
        def objective(n):
            return power_function(effect_size, int(n)) - power
        
        try:
            sample_size = optimize.brentq(objective, 4, 10000)
            sample_size = math.ceil(sample_size)
        except ValueError:
            sample_size = float('inf')
    
    else:
        power = power_function(effect_size, sample_size)
    
    results = {
        'test_type': 'Correlation power analysis',
        'effect_size': effect_size,
        'sample_size': sample_size,
        'power': power,
        'alpha': alpha,
        'alternative': alternative
    }
    
    return results

def perform_power_analysis(test_type='ttest', effect_size=None, sample_size=None, 
                         power=None, alpha=0.05, **kwargs):
    """
    Main function to perform power analysis for different statistical tests.
    
    Parameters:
    - test_type: Type of test ('ttest', 'chisquare', 'anova', 'correlation')
    - effect_size: Effect size for the test
    - sample_size: Sample size
    - power: Statistical power
    - alpha: Type I error rate
    - **kwargs: Additional test-specific parameters
    """
    
    try:
        if test_type == 'ttest':
            results = power_analysis_ttest(
                effect_size=effect_size, 
                sample_size=sample_size, 
                power=power, 
                alpha=alpha,
                test_type=kwargs.get('ttest_type', 'two_sample'),
                alternative=kwargs.get('alternative', 'two_sided')
            )
        
        elif test_type == 'chisquare':
            results = power_analysis_chisquare(
                effect_size=effect_size,
                sample_size=sample_size,
                power=power,
                alpha=alpha,
                df=kwargs.get('df', 1),
                test_type=kwargs.get('chi_type', 'independence')
            )
        
        elif test_type == 'anova':
            results = power_analysis_anova(
                effect_size=effect_size,
                sample_size=sample_size,
                power=power,
                alpha=alpha,
                groups=kwargs.get('groups', 3)
            )
        
        elif test_type == 'correlation':
            results = power_analysis_correlation(
                effect_size=effect_size,
                sample_size=sample_size,
                power=power,
                alpha=alpha,
                alternative=kwargs.get('alternative', 'two_sided')
            )
        
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def print_results(results, output_file=None):
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nStatistical Power Analysis Results:")
    print("=" * 80)
    print(f"Test type: {results['test_type']}")
    print(f"Significance level (alpha): {results['alpha']}")
    print("=" * 80)

    print("\nTEST DETAILS:")
    print("-" * 40)
    
    if 'ttest' in results['test_type'].lower():
        print("Analysis: Statistical power analysis for t-test")
        print("Effect size measure: Cohen's d")
        if 'alternative' in results:
            print(f"Alternative hypothesis: {results['alternative']}")
    
    elif 'chi-square' in results['test_type'].lower():
        print("Analysis: Statistical power analysis for chi-square test")
        print("Effect size measure: Cohen's w (or Cramer's V)")
        if 'degrees_of_freedom' in results:
            print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    
    elif 'anova' in results['test_type'].lower():
        print("Analysis: Statistical power analysis for one-way ANOVA")
        print("Effect size measure: Cohen's f")
        if 'groups' in results:
            print(f"Number of groups: {results['groups']}")
    
    elif 'correlation' in results['test_type'].lower():
        print("Analysis: Statistical power analysis for correlation")
        print("Effect size measure: Pearson's r")
        if 'alternative' in results:
            print(f"Alternative hypothesis: {results['alternative']}")

    print("\nPOWER ANALYSIS RESULTS:")
    print("-" * 40)
    
    if results['effect_size'] == float('inf'):
        print("Effect size: Unable to calculate (no solution found)")
    else:
        print(f"Effect size: {results['effect_size']:.4f}")
    
    if results['sample_size'] == float('inf'):
        print("Sample size: Unable to calculate (no solution found)")
    else:
        if 'sample_size_per_group' in results:
            print(f"Sample size per group: {results['sample_size_per_group']}")
            print(f"Total sample size: {results['total_sample_size']}")
        else:
            print(f"Sample size: {results['sample_size']}")
            if 'total_sample_size' in results:
                print(f"Total sample size: {results['total_sample_size']}")
    
    if results['power'] > 1:
        print("Statistical power: >0.9999 (essentially 1.0)")
    else:
        print(f"Statistical power: {results['power']:.4f}")

    print("\nEFFECT SIZE INTERPRETATION:")
    print("-" * 40)
    
    if results['effect_size'] != float('inf'):
        if 'ttest' in results['test_type'].lower():
            if results['effect_size'] < 0.2:
                interpretation = "negligible"
            elif results['effect_size'] < 0.5:
                interpretation = "small"
            elif results['effect_size'] < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            print(f"Cohen's d = {results['effect_size']:.3f} represents a {interpretation} effect size.")
        
        elif 'anova' in results['test_type'].lower():
            if results['effect_size'] < 0.1:
                interpretation = "small"
            elif results['effect_size'] < 0.25:
                interpretation = "medium"
            else:
                interpretation = "large"
            print(f"Cohen's f = {results['effect_size']:.3f} represents a {interpretation} effect size.")
        
        elif 'correlation' in results['test_type'].lower():
            if abs(results['effect_size']) < 0.1:
                interpretation = "negligible"
            elif abs(results['effect_size']) < 0.3:
                interpretation = "small"
            elif abs(results['effect_size']) < 0.5:
                interpretation = "medium"
            else:
                interpretation = "large"
            print(f"r = {results['effect_size']:.3f} represents a {interpretation} correlation.")
        
        elif 'chi-square' in results['test_type'].lower():
            if results['effect_size'] < 0.1:
                interpretation = "small"
            elif results['effect_size'] < 0.3:
                interpretation = "medium"
            else:
                interpretation = "large"
            print(f"w = {results['effect_size']:.3f} represents a {interpretation} effect size.")

    print("\nPOWER INTERPRETATION:")
    print("-" * 40)
    
    if results['power'] != float('inf') and results['power'] <= 1:
        if results['power'] < 0.8:
            power_level = "insufficient"
            recommendation = "Consider increasing sample size or effect size."
        elif results['power'] < 0.9:
            power_level = "adequate"
            recommendation = "Acceptable power for most research contexts."
        else:
            power_level = "high"
            recommendation = "Excellent power to detect the specified effect."
        
        print(f"Statistical power of {results['power']:.3f} is considered {power_level}.")
        print(f"Recommendation: {recommendation}")
    
    if results['power'] != float('inf') and results['power'] <= 1:
        beta = 1 - results['power']
        print(f"\nType II error rate (β): {beta:.4f}")
        print(f"This means there is a {beta*100:.1f}% chance of failing to detect a true effect of this size.")

    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Perform statistical power analysis for various tests.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate required sample size for t-test with medium effect size and 80% power
  python power_analysis.py ttest --effect_size 0.5 --power 0.8
  
  # Calculate power for chi-square test with given sample size and effect size
  python power_analysis.py chisquare --effect_size 0.3 --sample_size 100 --df 2
  
  # Calculate effect size needed for ANOVA with 3 groups, n=20 per group, power=0.9
  python power_analysis.py anova --sample_size 20 --power 0.9 --groups 3
        """
    )
    
    parser.add_argument('test_type', choices=['ttest', 'chisquare', 'anova', 'correlation'],
                       help='Type of statistical test for power analysis')
    
    # Core parameters (exactly one must be omitted)
    parser.add_argument('--effect_size', type=float, default=None,
                       help='Effect size (Cohen\'s d for t-test, f for ANOVA, r for correlation, w for chi-square)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size (per group for multi-group tests)')
    parser.add_argument('--power', type=float, default=None,
                       help='Statistical power (1 - β)')
    
    # Common parameters
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Type I error rate (significance level, default: 0.05)')
    
    # T-test specific parameters
    parser.add_argument('--ttest_type', choices=['one_sample', 'two_sample', 'paired'],
                       default='two_sample', help='Type of t-test (default: two_sample)')
    parser.add_argument('--alternative', choices=['two_sided', 'greater', 'less'],
                       default='two_sided', help='Alternative hypothesis (default: two_sided)')
    
    # Chi-square specific parameters
    parser.add_argument('--df', type=int, default=1,
                       help='Degrees of freedom for chi-square test (default: 1)')
    parser.add_argument('--chi_type', choices=['independence', 'goodness_of_fit'],
                       default='independence', help='Type of chi-square test (default: independence)')
    
    # ANOVA specific parameters
    parser.add_argument('--groups', type=int, default=3,
                       help='Number of groups for ANOVA (default: 3)')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    # Validate that exactly one of the main parameters is None
    params = [args.effect_size, args.sample_size, args.power]
    none_count = sum(1 for p in params if p is None)
    
    if none_count != 1:
        parser.error("Exactly one of --effect_size, --sample_size, or --power must be omitted")
    
    # Validate parameter ranges
    if args.alpha is not None and not (0 < args.alpha < 1):
        parser.error("Alpha must be between 0 and 1")
    
    if args.power is not None and not (0 < args.power < 1):
        parser.error("Power must be between 0 and 1")
    
    if args.sample_size is not None and args.sample_size <= 1:
        parser.error("Sample size must be greater than 1")
    
    kwargs = {
        'ttest_type': args.ttest_type,
        'alternative': args.alternative,
        'df': args.df,
        'chi_type': args.chi_type,
        'groups': args.groups
    }
    
    results = perform_power_analysis(
        test_type=args.test_type,
        effect_size=args.effect_size,
        sample_size=args.sample_size,
        power=args.power,
        alpha=args.alpha,
        **kwargs
    )
    
    print_results(results, args.output)

if __name__ == "__main__":
    main()