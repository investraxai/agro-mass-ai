"""
Model Training for Biomass Prediction Project

This script trains a linear regression model to predict biomass from NDVI values.
It includes data splitting, model training, evaluation, and interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_and_prepare_data(filename='sample_ndvi_biomass_data.csv'):
    """
    Load the generated dataset and prepare it for training.
    
    Args:
        filename (str): Path to the CSV file with NDVI and biomass data
        
    Returns:
        tuple: (X, y) where X is NDVI values and y is biomass values
    """
    print(" Loading and preparing data...")
    
    try:
        df = pd.read_csv(filename)
        print(f"   Loaded {len(df)} data points from {filename}")
    except FileNotFoundError:
        print(f"   Error: Could not find {filename}")
        print("   Please run sample_data_generator.py first to create the dataset.")
        return None, None
    
    # Extract features (X) and target (y)
    X = df['ndvi'].values.reshape(-1, 1)  # Reshape for sklearn
    y = df['biomass'].values
    
    print(f"   Feature shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X (numpy.ndarray): Feature data (NDVI values)
        y (numpy.ndarray): Target data (biomass values)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(" Splitting data into training and testing sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Testing set: {len(X_test)} samples")
    print(f"   Split ratio: {len(X_train)}:{len(X_test)} ({test_size*100}% for testing)")
    
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        
    Returns:
        sklearn.linear_model.LinearRegression: Trained model
    """
    print(" Training linear regression model...")
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get model parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    
    print(f"   Model trained successfully!")
    print(f"   Equation: Biomass = {slope:.2f} × NDVI + {intercept:.2f}")
    print(f"   Slope: {slope:.2f} (tons/hectare per NDVI unit)")
    print(f"   Intercept: {intercept:.2f} (tons/hectare at NDVI = 0)")
    
    return model

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Evaluate the trained model using various metrics.
    
    Args:
        model: Trained linear regression model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(" Evaluating model performance...")
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Calculate metrics for test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Calculate metrics for training set
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Store metrics
    metrics = {
        'test': {
            'mse': mse_test,
            'rmse': rmse_test,
            'mae': mae_test,
            'r2': r2_test
        },
        'train': {
            'mse': mse_train,
            'rmse': rmse_train,
            'r2': r2_train
        }
    }
    
    # Print results
    print("\n    Test Set Performance:")
    print(f"      Mean Squared Error (MSE): {mse_test:.2f}")
    print(f"      Root Mean Squared Error (RMSE): {rmse_test:.2f} tons/hectare")
    print(f"      Mean Absolute Error (MAE): {mae_test:.2f} tons/hectare")
    print(f"      R² Score: {r2_test:.3f}")
    
    print("\n    Training Set Performance:")
    print(f"      R² Score: {r2_train:.3f}")
    print(f"      RMSE: {rmse_train:.2f} tons/hectare")
    
    # Check for overfitting
    if r2_train - r2_test > 0.1:
        print("\n     Warning: Possible overfitting detected!")
        print("      Training R² is significantly higher than test R²")
    else:
        print("\n    Model generalization looks good!")
    
    return metrics

def interpret_model_results(model, metrics):
    """
    Provide interpretation of the model results in simple terms.
    
    Args:
        model: Trained linear regression model
        metrics (dict): Model evaluation metrics
    """
    print("\n Model Interpretation:")
    print("=" * 40)
    
    # Interpret R²
    r2 = metrics['test']['r2']
    if r2 >= 0.8:
        print(f"    Excellent fit! R² = {r2:.3f}")
        print("      The model explains most of the variation in biomass.")
    elif r2 >= 0.6:
        print(f"    Good fit! R² = {r2:.3f}")
        print("      The model explains a good portion of the variation in biomass.")
    elif r2 >= 0.4:
        print(f"    Moderate fit. R² = {r2:.3f}")
        print("      The model explains some variation, but there's room for improvement.")
    else:
        print(f"    Weak fit. R² = {r2:.3f}")
        print("      The model doesn't explain much variation. Consider other features.")
    
    # Interpret RMSE
    rmse = metrics['test']['rmse']
    print(f"\n    Prediction Accuracy:")
    print(f"      On average, predictions are off by {rmse:.1f} tons/hectare")
    print(f"      This represents {rmse/100*100:.1f}% of the typical biomass range")
    
    # Interpret slope
    slope = model.coef_[0]
    print(f"\n    Relationship Strength:")
    if slope > 0:
        print(f"      Positive relationship: Higher NDVI → Higher biomass")
        print(f"      For every 0.1 increase in NDVI, biomass increases by {slope*0.1:.1f} tons/hectare")
    else:
        print(f"      Negative relationship: Higher NDVI → Lower biomass")
        print(f"      This is unusual and might indicate data issues")
    
    # Practical implications
    print(f"\n    Practical Implications:")
    print("      • NDVI can be used to estimate biomass in the field")
    print("      • Higher NDVI values generally indicate more productive vegetation")
    print("      • The model provides a baseline for biomass monitoring")

def create_model_visualizations(model, X_train, y_train, X_test, y_test, metrics):
    """
    Create visualizations to understand the model performance.
    
    Args:
        model: Trained linear regression model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        metrics (dict): Model evaluation metrics
    """
    print(" Creating model performance visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Regression Model Performance', fontsize=16)
    
    # 1. Training data with regression line
    axes[0, 0].scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data')
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    axes[0, 0].plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
    axes[0, 0].set_title(f'Training Data & Model (R² = {metrics["train"]["r2"]:.3f})')
    axes[0, 0].set_xlabel('NDVI Value')
    axes[0, 0].set_ylabel('Biomass (tons/hectare)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test data with regression line
    axes[0, 1].scatter(X_test, y_test, alpha=0.6, color='green', label='Test Data')
    axes[0, 1].plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
    axes[0, 1].set_title(f'Test Data & Model (R² = {metrics["test"]["r2"]:.3f})')
    axes[0, 1].set_xlabel('NDVI Value')
    axes[0, 1].set_ylabel('Biomass (tons/hectare)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Predicted vs Actual (Test Set)
    y_pred_test = model.predict(X_test)
    axes[1, 0].scatter(y_test, y_pred_test, alpha=0.6, color='purple')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_title('Predicted vs Actual Values (Test Set)')
    axes[1, 0].set_xlabel('Actual Biomass (tons/hectare)')
    axes[1, 0].set_ylabel('Predicted Biomass (tons/hectare)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals plot
    residuals = y_test - y_pred_test
    axes[1, 1].scatter(y_pred_test, residuals, alpha=0.6, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Residuals Plot (Test Set)')
    axes[1, 1].set_xlabel('Predicted Biomass (tons/hectare)')
    axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'model_performance.png'")
    plt.show()

def save_model_results(model, metrics, filename='model_results.txt'):
    """
    Save model results and parameters to a text file.
    
    Args:
        model: Trained linear regression model
        metrics (dict): Model evaluation metrics
        filename (str): Output filename
    """
    print(f" Saving model results to {filename}...")
    
    with open(filename, 'w') as f:
        f.write("BIOMASS PREDICTION MODEL RESULTS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("MODEL PARAMETERS:\n")
        f.write(f"Slope (coefficient): {model.coef_[0]:.4f}\n")
        f.write(f"Intercept: {model.intercept_:.4f}\n")
        f.write(f"Equation: Biomass = {model.coef_[0]:.4f} × NDVI + {model.intercept_:.4f}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("Test Set:\n")
        f.write(f"  R² Score: {metrics['test']['r2']:.4f}\n")
        f.write(f"  RMSE: {metrics['test']['rmse']:.4f} tons/hectare\n")
        f.write(f"  MAE: {metrics['test']['mae']:.4f} tons/hectare\n")
        f.write(f"  MSE: {metrics['test']['mse']:.4f}\n\n")
        
        f.write("Training Set:\n")
        f.write(f"  R² Score: {metrics['train']['r2']:.4f}\n")
        f.write(f"  RMSE: {metrics['train']['rmse']:.4f} tons/hectare\n\n")
        
        f.write("INTERPRETATION:\n")
        if metrics['test']['r2'] >= 0.8:
            f.write("Excellent model fit - explains most variation in biomass\n")
        elif metrics['test']['r2'] >= 0.6:
            f.write("Good model fit - explains good portion of variation\n")
        elif metrics['test']['r2'] >= 0.4:
            f.write("Moderate model fit - some room for improvement\n")
        else:
            f.write("Weak model fit - consider additional features\n")
    
    print(f"   Results saved to {filename}")

def main():
    """
    Main function to run the complete model training pipeline.
    """
    print(" Starting Model Training for Biomass Prediction")
    print("=" * 60)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Interpret results
    interpret_model_results(model, metrics)
    
    # Create visualizations
    create_model_visualizations(model, X_train, y_train, X_test, y_test, metrics)
    
    # Save results
    save_model_results(model, metrics)
    
    print("\n Model training complete!")
    print("You now have a working biomass prediction model!")
    print("\nNext steps:")
    print("  • Use the model to predict biomass for new NDVI values")
    print("  • Try different machine learning algorithms")
    print("  • Incorporate additional features (temperature, rainfall, etc.)")

if __name__ == "__main__":
    main()

