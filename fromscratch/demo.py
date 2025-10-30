#!/usr/bin/env python3
"""
Simple Demo Script for Biomass Prediction Project

This script demonstrates the key concepts in a concise way.
Run this to see the project in action quickly!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    print(" BIOMASS PREDICTION PROJECT - QUICK DEMO")
    print("=" * 50)
    
    # Step 1: Generate simple data
    print("\n1️ Generating sample data...")
    np.random.seed(42)  # For reproducibility
    
    # Create NDVI values (vegetation health indicator)
    ndvi = np.random.uniform(-0.5, 1.0, 200)
    
    # Create biomass values that correlate with NDVI
    # Higher NDVI = higher biomass, but with some noise
    biomass = 50 + 100 * (ndvi + 0.5) + np.random.normal(0, 15, 200)
    biomass = np.maximum(biomass, 0)  # Ensure positive values
    
    print(f"   Created {len(ndvi)} data points")
    print(f"   NDVI range: {ndvi.min():.2f} to {ndvi.max():.2f}")
    print(f"   Biomass range: {biomass.min():.1f} to {biomass.max():.1f} tons/hectare")
    print(f"   Correlation: {np.corrcoef(ndvi, biomass)[0,1]:.3f}")
    
    # Step 2: Prepare data for modeling
    print("\n2️ Preparing data for modeling...")
    X = ndvi.reshape(-1, 1)  # Features (NDVI)
    y = biomass              # Target (biomass)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Testing set: {len(X_test)} samples")
    
    # Step 3: Train the model
    print("\n3️ Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get model parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"   Model equation: Biomass = {slope:.1f} × NDVI + {intercept:.1f}")
    print(f"   This means: for every 0.1 increase in NDVI, biomass increases by {slope*0.1:.1f} tons/hectare")
    
    # Step 4: Evaluate the model
    print("\n4️ Evaluating model performance...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   R² Score: {r2:.3f}")
    if r2 >= 0.7:
        print("    Excellent model fit!")
    elif r2 >= 0.5:
        print("    Good model fit!")
    else:
        print("    Model could be improved")
    
    # Step 5: Make predictions
    print("\n5️ Making predictions...")
    example_ndvi = np.array([-0.3, 0.2, 0.7, 0.9])
    predictions = model.predict(example_ndvi.reshape(-1, 1))
    
    print("   Example predictions:")
    print("   NDVI Value | Predicted Biomass | Vegetation Type")
    print("   " + "-" * 45)
    
    for ndvi_val, pred in zip(example_ndvi, predictions):
        if ndvi_val < 0:
            vtype = "Water/Barren"
        elif ndvi_val < 0.3:
            vtype = "Sparse"
        elif ndvi_val < 0.6:
            vtype = "Moderate"
        else:
            vtype = "Dense"
        
        print(f"   {ndvi_val:>8.1f} | {pred:>16.1f} | {vtype}")
    
    # Step 6: Create visualization
    print("\n6️ Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(X_train, y_train, alpha=0.6, color='blue', s=30, label='Training Data')
    
    # Plot test data
    plt.scatter(X_test, y_test, alpha=0.6, color='green', s=30, label='Test Data')
    
    # Plot prediction line
    ndvi_range = np.linspace(-0.5, 1, 100).reshape(-1, 1)
    biomass_range = model.predict(ndvi_range)
    plt.plot(ndvi_range, biomass_range, 'r-', linewidth=3, label='Model Prediction')
    
    # Highlight example predictions
    plt.scatter(example_ndvi, predictions, c='red', s=200, marker='s', 
               label='Example Predictions', zorder=5)
    
    # Add annotations
    for i, (ndvi_val, biomass_val) in enumerate(zip(example_ndvi, predictions)):
        plt.annotate(f'NDVI={ndvi_val:.1f}\nBiomass={biomass_val:.1f}', 
                     xy=(ndvi_val, biomass_val), xytext=(10, 10),
                     textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('NDVI Value (Vegetation Health)')
    plt.ylabel('Biomass (tons/hectare)')
    plt.title('Biomass Prediction Model\n(Red squares show example predictions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'demo_visualization.png'")
    plt.show()
    
    # Step 7: Summary
    print("\n PROJECT SUMMARY")
    print("=" * 30)
    print(" What you accomplished:")
    print("   • Generated realistic NDVI and biomass data")
    print("   • Built a linear regression model")
    print("   • Evaluated model performance (R² = {:.3f})".format(r2))
    print("   • Made predictions for new NDVI values")
    print("   • Created a visualization")
    
    print("\n Key concepts learned:")
    print("   • NDVI measures vegetation health (-1 to 1)")
    print("   • Higher NDVI generally means more biomass")
    print("   • Linear regression can model this relationship")
    print("   • Model evaluation shows how well it works")
    
    print("\n Next steps:")
    print("   • Try different machine learning algorithms")
    print("   • Use real satellite data")
    print("   • Add more features (temperature, rainfall)")
    print("   • Apply to real agricultural areas")
    
    print("\n The model equation: Biomass = {:.1f} × NDVI + {:.1f}".format(slope, intercept))
    print("   Use this to predict biomass for any NDVI value!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error: {str(e)}")
        print("Make sure you have the required packages installed:")
        print("pip install numpy pandas matplotlib scikit-learn")

