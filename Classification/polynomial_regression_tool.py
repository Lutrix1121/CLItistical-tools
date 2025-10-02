import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import warnings
from scipy import stats

def perform_polynomial_regression(csv_file, target_column, predictor_columns=None, 
                                 degree=2, test_size=0.2, random_state=42, 
                                 standardize=True, separator=',', fit_intercept=True,
                                 interaction_only=False, include_bias=True):
    """
    Performs polynomial regression analysis on data from a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    target_column (str): Name of the target/dependent variable column.
    predictor_columns (list, optional): List of predictor/independent variable column names. 
                                       If None, uses all numeric columns except target.
    degree (int, optional): Degree of the polynomial features (default: 2).
    test_size (float, optional): Proportion of data to use for testing (default: 0.2).
    random_state (int, optional): Random state for reproducibility (default: 42).
    standardize (bool, optional): Whether to standardize features (default: True).
    separator (str, optional): Separator used in CSV file (default: ',').
    fit_intercept (bool, optional): Whether to calculate intercept (default: True).
    interaction_only (bool, optional): If True, only interaction features are produced (default: False).
    include_bias (bool, optional): If True, include a bias column (default: True).
    
    Returns:
    dict: Results containing model performance metrics and statistics.
    """
    try:
        data = pd.read_csv(csv_file, sep=separator)
        print(f"Loaded data with shape: {data.shape}")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")
        
        if data[target_column].isnull().any():
            print(f"Warning: Found {data[target_column].isnull().sum()} missing values in target column. Removing these rows.")
            data = data.dropna(subset=[target_column])
        
        y = data[target_column].copy()
        
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Target variable must be numeric. Found type: {y.dtype}")
        
        if predictor_columns is None:
            # Use all numeric columns except target
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
            predictor_columns = numeric_columns
            print(f"Using all numeric columns as predictors: {predictor_columns}")
        else:
            missing_columns = [col for col in predictor_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Predictor columns do not exist in dataset: {missing_columns}")
        
        if not predictor_columns:
            raise ValueError("No predictor columns available for analysis.")
        
        X = data[predictor_columns].copy()
        
        if X.isnull().any().any():
            missing_info = X.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            print(f"Warning: Found missing values in predictor columns:\n{missing_cols}")
            print("Removing rows with missing predictor values.")
            X = X.dropna()
            y = y[X.index]
        
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            print(f"Encoding categorical columns: {categorical_columns}")
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            predictor_columns = X.columns.tolist()
        
        # Check final data size
        if len(X) < 30:
            raise ValueError("Insufficient data for analysis. Need at least 30 observations after cleaning.")
        
        print(f"Initial feature matrix shape: {X.shape}")
        print(f"Target variable statistics: Mean={y.mean():.4f}, Std={y.std():.4f}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Standardize features if requested (before polynomial transformation)
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Create polynomial features
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        
        # Get feature names
        if hasattr(poly, 'get_feature_names_out'):
            poly_feature_names = poly.get_feature_names_out(predictor_columns)
        else:
            poly_feature_names = poly.get_feature_names(predictor_columns)
        
        print(f"Polynomial feature matrix shape: {X_train_poly.shape}")
        print(f"Number of polynomial features created: {len(poly_feature_names)}")
        
        # Fit linear regression model on polynomial features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        
        # Calculate performance metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        n_train = len(y_train)
        n_features = X_train_poly.shape[1]
        adjusted_r2_train = 1 - (1 - train_r2) * (n_train - 1) / (n_train - n_features - 1)
        
        n_test = len(y_test)
        adjusted_r2_test = 1 - (1 - test_r2) * (n_test - 1) / (n_test - n_features - 1)
        
        coefficients = model.coef_
        intercept = model.intercept_ if fit_intercept else 0.0
        
        residuals = y_test - y_pred_test
        
        # Statistical tests on residuals
        # Shapiro-Wilk test for normality (if sample size <= 5000)
        normality_test = None
        if len(residuals) <= 5000:
            stat, p_value = stats.shapiro(residuals)
            normality_test = {'statistic': stat, 'p_value': p_value}
        
        # Durbin-Watson test for autocorrelation
        def durbin_watson(residuals):
            diff = np.diff(residuals)
            return np.sum(diff**2) / np.sum(residuals**2)
        
        dw_statistic = durbin_watson(residuals)
        
        feature_importance = pd.DataFrame({
            'feature': poly_feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Calculate confidence intervals for coefficients (approximate)
        residual_std = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Prepare results
        results = {
            'model': model,
            'poly_transformer': poly,
            'scaler': scaler,
            'original_feature_columns': predictor_columns,
            'poly_feature_names': poly_feature_names,
            'target_column': target_column,
            'degree': degree,
            'interaction_only': interaction_only,
            'intercept': intercept,
            'coefficients': coefficients,
            'feature_importance': feature_importance,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'adjusted_r2_train': adjusted_r2_train,
            'adjusted_r2_test': adjusted_r2_test,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'residuals': residuals,
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'normality_test': normality_test,
            'durbin_watson': dw_statistic,
            'residual_std': residual_std,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'n_original_features': len(predictor_columns),
            'n_poly_features': len(poly_feature_names),
            'standardized': standardize,
            'fit_intercept': fit_intercept,
            'target_mean': y.mean(),
            'target_std': y.std()
        }
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def print_results(results, output_file="polynomial_regression_results.txt"):
    """Print formatted results of polynomial regression analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        import sys
        import os
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nPolynomial Regression Analysis Results:")
    print("=" * 80)
    print(f"Target variable: {results['target_column']}")
    print(f"Polynomial degree: {results['degree']}")
    print(f"Interaction only: {results['interaction_only']}")
    print(f"Number of original features: {results['n_original_features']}")
    print(f"Number of polynomial features: {results['n_poly_features']}")
    print(f"Training set size: {results['train_size']}")
    print(f"Test set size: {results['test_size']}")
    print(f"Features standardized: {results['standardized']}")
    print(f"Fit intercept: {results['fit_intercept']}")
    print("=" * 80)
    
    print("\nMODEL PERFORMANCE:")
    print("-" * 40)
    print(f"Training R2: {results['train_r2']:.4f}")
    print(f"Test R2: {results['test_r2']:.4f}")
    print(f"Training Adjusted R2: {results['adjusted_r2_train']:.4f}")
    print(f"Test Adjusted R2: {results['adjusted_r2_test']:.4f}")
    print(f"Training RMSE: {results['train_rmse']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    print(f"Training MAE: {results['train_mae']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    
    r2_diff = results['train_r2'] - results['test_r2']
    if r2_diff > 0.1:
        print("\nWARNING: Significant difference between training and test R2.")
        print("This may indicate overfitting. Consider:")
        print("- Reducing polynomial degree")
        print("- Regularization (Ridge/Lasso regression)")
        print("- Increasing training data")
    
    print("\nMODEL COEFFICIENTS:")
    print("-" * 40)
    if results['fit_intercept']:
        print(f"Intercept: {results['intercept']:.4f}")
    
    print("\nTOP 10 MOST IMPORTANT POLYNOMIAL FEATURES:")
    print("-" * 70)
    print(f"{'Feature':<45} {'Coefficient':<15}")
    print("-" * 70)
    
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        feature_name = row['feature']
        if len(feature_name) > 44:
            feature_name = feature_name[:41] + "..."
        print(f"{feature_name:<45} {row['coefficient']:<15.4f}")
    
    print("\nRESIDUAL ANALYSIS:")
    print("-" * 40)
    residuals = results['residuals']
    print(f"Residual Mean: {np.mean(residuals):.6f}")
    print(f"Residual Std: {results['residual_std']:.4f}")
    print(f"Residual Min: {np.min(residuals):.4f}")
    print(f"Residual Max: {np.max(residuals):.4f}")
    
    print("\nSTATISTICAL TESTS:")
    print("-" * 40)
    
    if results['normality_test'] is not None:
        norm_test = results['normality_test']
        print(f"Shapiro-Wilk Normality Test:")
        print(f"  Statistic: {norm_test['statistic']:.4f}")
        print(f"  P-value: {norm_test['p_value']:.4f}")
        if norm_test['p_value'] < 0.05:
            print("  Result: Residuals are NOT normally distributed (p < 0.05)")
        else:
            print("  Result: Residuals appear normally distributed (p >= 0.05)")
    else:
        print("Shapiro-Wilk test skipped (sample size > 5000)")
    
    dw = results['durbin_watson']
    print(f"\nDurbin-Watson Test: {dw:.4f}")
    if dw < 1.5:
        print("  Result: Positive autocorrelation detected")
    elif dw > 2.5:
        print("  Result: Negative autocorrelation detected")
    else:
        print("  Result: No significant autocorrelation")
    
    print("\nMODEL INTERPRETATION:")
    print("-" * 40)
    most_important = results['feature_importance'].iloc[0]
    print(f"Most important polynomial feature: {most_important['feature']}")
    print(f"Coefficient: {most_important['coefficient']:.4f}")
    
    if most_important['coefficient'] > 0:
        direction = "increases"
    else:
        direction = "decreases"
    
    print(f"This feature {direction} the target by {abs(most_important['coefficient']):.4f} units per unit change")
    
    r2_test = results['test_r2']
    print(f"\nTest R2 value: {r2_test:.4f}")
    if r2_test > 0.8:
        print("Model shows excellent predictive performance (R2 > 0.8)")
    elif r2_test > 0.6:
        print("Model shows good predictive performance (R2 > 0.6)")
    elif r2_test > 0.4:
        print("Model shows moderate predictive performance (R2 > 0.4)")
    else:
        print("Model shows limited predictive performance. Consider:")
        print("- Adjusting polynomial degree")
        print("- Adding more relevant features")
        print("- Feature engineering")
        print("- Different modeling approaches")
    
    print("\nPOLYNOMIAL REGRESSION INSIGHTS:")
    print("-" * 40)
    print(f"The model uses degree {results['degree']} polynomial features.")
    print(f"Total features expanded from {results['n_original_features']} to {results['n_poly_features']}.")
    
    if results['degree'] > 3:
        print("\nNote: High-degree polynomials may lead to overfitting.")
        print("Monitor the difference between training and test performance.")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Polynomial Regression analysis for data from CSV files')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target/dependent variable column')
    parser.add_argument('--predictor_columns', type=str, nargs='+', default=None,
                        help='Names of predictor/independent variable columns (if not specified, uses all numeric columns)')
    parser.add_argument('--degree', type=int, default=2,
                        help='Degree of the polynomial features (default: 2)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--no_standardize', action='store_true',
                        help='Do not standardize features (default: standardize)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in CSV file (default: ",")')
    parser.add_argument('--no_intercept', action='store_true',
                        help='Do not fit intercept term (default: fit intercept)')
    parser.add_argument('--interaction_only', action='store_true',
                        help='Only interaction features (no powers) (default: False)')
    parser.add_argument('--no_bias', action='store_true',
                        help='Do not include bias column in polynomial features (default: include bias)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    results = perform_polynomial_regression(
        args.csv_file, args.target_column, args.predictor_columns,
        args.degree, args.test_size, args.random_state, 
        not args.no_standardize, args.separator, not args.no_intercept,
        args.interaction_only, not args.no_bias
    )
    
    print_results(results, args.output)

if __name__ == "__main__":
    main()
