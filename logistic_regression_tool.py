import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import warnings

def perform_logistic_regression(csv_file, target_column, predictor_columns=None, 
                               test_size=0.2, random_state=42, standardize=True,
                               separator=';', max_iter=1000, solver='lbfgs'):
    """
    Performs logistic regression analysis on data from a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    target_column (str): Name of the target/dependent variable column.
    predictor_columns (list, optional): List of predictor/independent variable column names. 
                                       If None, uses all numeric columns except target.
    test_size (float, optional): Proportion of data to use for testing (default: 0.2).
    random_state (int, optional): Random state for reproducibility (default: 42).
    standardize (bool, optional): Whether to standardize features (default: True).
    separator (str, optional): Separator used in CSV file (default: ',').
    max_iter (int, optional): Maximum number of iterations for solver (default: 1000).
    solver (str, optional): Algorithm for optimization problem (default: 'lbfgs').
    
    Returns:
    dict: Results containing model performance metrics and statistics.
    """
    try:
        # Load data from CSV file
        data = pd.read_csv(csv_file, sep=separator)
        print(f"Loaded data with shape: {data.shape}")
        
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")
        
        # Handle missing values in target column
        if data[target_column].isnull().any():
            print(f"Warning: Found {data[target_column].isnull().sum()} missing values in target column. Removing these rows.")
            data = data.dropna(subset=[target_column])
        
        # Prepare target variable
        y = data[target_column].copy()
        
        # Encode target variable if it's not numeric
        label_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print(f"Target variable encoded. Classes: {label_encoder.classes_}")
        
        # Check if target is binary
        unique_values = np.unique(y)
        if len(unique_values) != 2:
            raise ValueError(f"Logistic regression requires a binary target variable. Found {len(unique_values)} unique values: {unique_values}")
        
        # Prepare predictor variables
        if predictor_columns is None:
            # Use all numeric columns except target
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
            predictor_columns = numeric_columns
            print(f"Using all numeric columns as predictors: {predictor_columns}")
        else:
            # Check if specified predictor columns exist
            missing_columns = [col for col in predictor_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Predictor columns do not exist in dataset: {missing_columns}")
        
        if not predictor_columns:
            raise ValueError("No predictor columns available for analysis.")
        
        # Prepare feature matrix
        X = data[predictor_columns].copy()
        
        # Handle missing values in predictors
        if X.isnull().any().any():
            missing_info = X.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            print(f"Warning: Found missing values in predictor columns:\n{missing_cols}")
            print("Removing rows with missing predictor values.")
            X = X.dropna()
            y = y[X.index]
        
        # Encode categorical predictors
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            print(f"Encoding categorical columns: {categorical_columns}")
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            predictor_columns = X.columns.tolist()
        
        # Check final data size
        if len(X) < 30:
            raise ValueError("Insufficient data for analysis. Need at least 10 observations after cleaning.")
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Target variable distribution: {np.bincount(y)}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features if requested
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Fit logistic regression model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(
                max_iter=max_iter, 
                random_state=random_state,
                solver=solver
            )
            model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate performance metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_auc = roc_auc_score(y_train, y_prob_train)
        test_auc = roc_auc_score(y_test, y_prob_test)
        
        # Confusion matrix
        cm_test = confusion_matrix(y_test, y_pred_test)
        
        # Classification report
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Get feature coefficients and their significance
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Calculate odds ratios
        odds_ratios = np.exp(coefficients)
        
        # Create feature importance summary
        feature_importance = pd.DataFrame({
            'feature': predictor_columns,
            'coefficient': coefficients,
            'odds_ratio': odds_ratios,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Prepare results
        results = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': predictor_columns,
            'target_column': target_column,
            'intercept': intercept,
            'coefficients': coefficients,
            'odds_ratios': odds_ratios,
            'feature_importance': feature_importance,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'confusion_matrix': cm_test,
            'classification_report': class_report,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'n_features': len(predictor_columns),
            'standardized': standardize,
            'solver': solver,
            'max_iter': max_iter
        }
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def print_results(results, output_file=None):
    """Print formatted results of logistic regression analysis."""
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

    print("\nLogistic Regression Analysis Results:")
    print("=" * 80)
    print(f"Target variable: {results['target_column']}")
    print(f"Number of features: {results['n_features']}")
    print(f"Training set size: {results['train_size']}")
    print(f"Test set size: {results['test_size']}")
    print(f"Features standardized: {results['standardized']}")
    print(f"Solver: {results['solver']}")
    print("=" * 80)
    
    # Model Performance
    print("\nMODEL PERFORMANCE:")
    print("-" * 40)
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Training AUC-ROC: {results['train_auc']:.4f}")
    print(f"Test AUC-ROC: {results['test_auc']:.4f}")
    
    # Confusion Matrix
    print("\nCONFUSION MATRIX (Test Set):")
    print("-" * 40)
    cm = results['confusion_matrix']
    print(f"True Negative: {cm[0,0]:4d}  |  False Positive: {cm[0,1]:4d}")
    print(f"False Negative: {cm[1,0]:4d}  |  True Positive: {cm[1,1]:4d}")
    
    # Classification Metrics
    print("\nCLASSIFICATION METRICS (Test Set):")
    print("-" * 40)
    cr = results['classification_report']
    for class_label in ['0', '1']:
        if class_label in cr:
            print(f"Class {class_label}:")
            print(f"  Precision: {cr[class_label]['precision']:.4f}")
            print(f"  Recall: {cr[class_label]['recall']:.4f}")
            print(f"  F1-Score: {cr[class_label]['f1-score']:.4f}")
    
    print(f"\nMacro Average F1-Score: {cr['macro avg']['f1-score']:.4f}")
    print(f"Weighted Average F1-Score: {cr['weighted avg']['f1-score']:.4f}")
    
    # Model Coefficients
    print("\nMODEL COEFFICIENTS:")
    print("-" * 40)
    print(f"Intercept: {results['intercept']:.4f}")
    
    print("\nFEATURE IMPORTANCE (Top 10):")
    print("-" * 60)
    print(f"{'Feature':<25} {'Coefficient':<12} {'Odds Ratio':<12}")
    print("-" * 60)
    
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        print(f"{row['feature']:<25} {row['coefficient']:<12.4f} {row['odds_ratio']:<12.4f}")
    
    # Model Interpretation
    print("\nMODEL INTERPRETATION:")
    print("-" * 40)
    most_important = results['feature_importance'].iloc[0]
    if most_important['coefficient'] > 0:
        direction = "increases"
    else:
        direction = "decreases"
    
    print(f"Most important feature: {most_important['feature']}")
    print(f"This feature {direction} the log-odds by {abs(most_important['coefficient']):.4f}")
    print(f"Odds ratio: {most_important['odds_ratio']:.4f}")
    
    if results['test_accuracy'] > 0.8:
        print("\nThe model shows good predictive performance.")
    elif results['test_accuracy'] > 0.7:
        print("\nThe model shows moderate predictive performance.")
    else:
        print("\nThe model shows limited predictive performance. Consider:")
        print("- Adding more relevant features")
        print("- Feature engineering")
        print("- Different modeling approaches")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Logistic Regression analysis for data from CSV files')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target/dependent variable column')
    parser.add_argument('--predictor_columns', type=str, nargs='+', default=None,
                        help='Names of predictor/independent variable columns (if not specified, uses all numeric columns)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--no_standardize', action='store_true',
                        help='Do not standardize features (default: standardize)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in CSV file (default: ",")')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum number of iterations for solver (default: 1000)')
    parser.add_argument('--solver', type=str, default='lbfgs',
                        choices=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                        help='Algorithm for optimization problem (default: "lbfgs")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    # Perform logistic regression
    results = perform_logistic_regression(
        args.csv_file, args.target_column, args.predictor_columns,
        args.test_size, args.random_state, not args.no_standardize,
        args.separator, args.max_iter, args.solver
    )
    
    # Display results
    print_results(results, args.output)

if __name__ == "__main__":
    main()
