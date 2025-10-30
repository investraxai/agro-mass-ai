"""
Main Biomass Prediction Project Script

This script orchestrates the complete workflow:
1. Generate sample NDVI and biomass data
2. Train a linear regression model
3. Evaluate model performance
4. Create comprehensive visualizations
5. Demonstrate predictions

Run this script to execute the entire project pipeline.
"""

import os
import sys
import time
from datetime import datetime

def print_header():
    """Print a beautiful project header."""
    print("" + "="*60 + "")
    print("    BIOMASS PREDICTION PROJECT - FROM SCRATCH    ")
    print("" + "="*60 + "")
    print(" Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(" Goal: Predict biomass using NDVI satellite data")
    print(" Method: Linear Regression with synthetic data")
    print("" + "="*60 + "\n")

def check_dependencies():
    """Check if required packages are installed."""
    print(" Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scikit-learn', 
        'seaborn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"    {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   {package} - MISSING")
    
    if missing_packages:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("   Please install them using: pip install -r requirements.txt")
        return False
    
    print("    All dependencies are available!")
    return True

def run_data_generation():
    """Run the data generation step."""
    print("\n STEP 1: Generating Sample Data")
    print("-" * 40)
    
    try:
        # Import and run data generation
        from sample_data_generator import create_sample_dataset, visualize_data_distribution
        
        print("   Creating synthetic NDVI and biomass dataset...")
        df = create_sample_dataset(n_samples=1000)
        
        print("   Creating initial data visualizations...")
        visualize_data_distribution(df)
        
        print("    Data generation complete!")
        return True
        
    except Exception as e:
        print(f"    Error in data generation: {str(e)}")
        return False

def run_model_training():
    """Run the model training step."""
    print("\n STEP 2: Training the Model")
    print("-" * 40)
    
    try:
        # Import and run model training
        from model_training import main as train_model
        
        print("   Training linear regression model...")
        train_model()
        
        print("    Model training complete!")
        return True
        
    except Exception as e:
        print(f"    Error in model training: {str(e)}")
        return False

def run_visualizations():
    """Run the visualization step."""
    print("\n STEP 3: Creating Visualizations")
    print("-" * 40)
    
    try:
        # Import and run visualizations
        from visualizations import main as create_viz
        
        print("   Creating comprehensive visualizations...")
        create_viz()
        
        print("    Visualizations complete!")
        return True
        
    except Exception as e:
        print(f"    Error in visualizations: {str(e)}")
        return False

def demonstrate_predictions():
    """Demonstrate how to use the trained model for predictions."""
    print("\n STEP 4: Making Predictions")
    print("-" * 40)
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Load the data and train a simple model
        df = pd.read_csv('sample_ndvi_biomass_data.csv')
        model = LinearRegression()
        X = df['ndvi'].values.reshape(-1, 1)
        y = df['biomass'].values
        model.fit(X, y)
        
        print("   Model equation: Biomass = {:.2f} × NDVI + {:.2f}".format(
            model.coef_[0], model.intercept_0))
        
        # Make some example predictions
        example_ndvi = np.array([-0.5, 0, 0.3, 0.6, 0.9])
        predictions = model.predict(example_ndvi.reshape(-1, 1))
        
        print("\n   📊 Example Predictions:")
        print("   NDVI Value | Predicted Biomass | Vegetation Type")
        print("   " + "-" * 50)
        
        for ndvi, pred in zip(example_ndvi, predictions):
            if ndvi < 0:
                vtype = "Water/Barren"
            elif ndvi < 0.3:
                vtype = "Sparse"
            elif ndvi < 0.6:
                vtype = "Moderate"
            else:
                vtype = "Dense"
            
            print(f"   {ndvi:>8.1f} | {pred:>16.1f} | {vtype}")
        
        print("\n   💡 How to use this model:")
        print("   • For any NDVI value, multiply by {:.2f} and add {:.2f}".format(
            model.coef_[0], model.intercept_0))
        print("   • Higher NDVI values generally predict higher biomass")
        print("   • The model works best for NDVI values between -0.5 and 1.0")
        
        return True
        
    except Exception as e:
        print(f"    Error in predictions: {str(e)}")
        return False

def show_project_summary():
    """Show a summary of what was accomplished."""
    print("\n PROJECT SUMMARY")
    print("=" * 50)
    
    # Check what files were created
    created_files = []
    expected_files = [
        'sample_ndvi_biomass_data.csv',
        'data_distribution.png',
        'model_performance.png',
        'model_results.txt',
        'comprehensive_visualizations.png',
        'prediction_examples.png',
        'data_quality_analysis.png'
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            created_files.append(f" {file}")
        else:
            created_files.append(f" {file}")
    
    print("   Generated Files:")
    for file_status in created_files:
        print(f"   {file_status}")
    
    print("\n    What You've Accomplished:")
    print("   • Created realistic synthetic NDVI and biomass data")
    print("   • Trained a linear regression model")
    print("   • Evaluated model performance with multiple metrics")
    print("   • Generated comprehensive visualizations")
    print("   • Learned how to make predictions with the model")
    
    print("\n    Key Concepts Learned:")
    print("   • NDVI measures vegetation health and density")
    print("   • Biomass correlates with NDVI values")
    print("   • Linear regression can model this relationship")
    print("   • Model evaluation requires multiple metrics")
    print("   • Visualizations help understand data and results")
    
    print("\n    Next Steps:")
    print("   • Try different machine learning algorithms")
    print("   • Use real satellite data from Landsat or Sentinel")
    print("   • Add more features (temperature, rainfall, soil type)")
    print("   • Apply to specific regions or crop types")
    print("   • Validate with ground truth measurements")

def main():
    """Main function to run the complete project pipeline."""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print("\n Please install missing dependencies and try again.")
        return
    
    print("\n Starting Biomass Prediction Project Pipeline...")
    print("   This will take a few minutes to complete.\n")
    
    start_time = time.time()
    
    # Step 1: Data Generation
    if not run_data_generation():
        print("\n Pipeline failed at data generation step.")
        return
    
    # Step 2: Model Training
    if not run_model_training():
        print("\n Pipeline failed at model training step.")
        return
    
    # Step 3: Visualizations
    if not run_visualizations():
        print("\n Pipeline failed at visualization step.")
        return
    
    # Step 4: Predictions
    if not demonstrate_predictions():
        print("\n Pipeline failed at prediction step.")
        return
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"\n⏱  Total execution time: {total_time:.1f} seconds")
    
    # Show summary
    show_project_summary()
    
    print("\n CONGRATULATIONS!")
    print("   You've successfully completed the Biomass Prediction Project!")
    print("   Check the generated files and visualizations to explore your results.")
    print("\n Happy learning and exploring!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Project interrupted by user.")
        print("   You can resume by running individual scripts:")
        print("   • python sample_data_generator.py")
        print("   • python model_training.py")
        print("   • python visualizations.py")
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        print("   Please check the error message and try again.")