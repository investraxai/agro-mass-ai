# Biomass Prediction Project

This project demonstrates how to predict biomass using NDVI (Normalized Difference Vegetation Index) data from satellite imagery. It's designed to be beginner-friendly and educational.

## What You'll Learn

1. **NDVI Data Generation**: Create synthetic NDVI data that mimics satellite imagery
2. **Ground Truth Data**: Generate realistic biomass measurements to match the NDVI data
3. **Linear Regression Model**: Build a simple machine learning model to predict biomass from NDVI
4. **Visualization**: Create clear plots showing the relationship between NDVI and biomass
5. **Model Evaluation**: Test your model and understand accuracy metrics

## Project Structure

```
fromscratch/
├── requirements.txt          # Python dependencies
├── biomass_prediction.py     # Main script with all functionality
├── sample_data_generator.py  # Generate synthetic NDVI and biomass data
├── model_training.py         # Train and evaluate the linear regression model
├── visualizations.py         # Create plots and charts
└── README.md                 # This file
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete project:
   ```bash
   python biomass_prediction.py
   ```

3. Or run individual components:
   ```bash
   python sample_data_generator.py    # Generate data
   python model_training.py           # Train model
   python visualizations.py           # Create plots
   ```

## Understanding the Concepts

### NDVI (Normalized Difference Vegetation Index)
- NDVI measures vegetation health and density
- Values range from -1 to 1
- Higher values indicate healthier, denser vegetation
- Formula: NDVI = (NIR - Red) / (NIR + Red)

### Biomass
- Total mass of living organisms in a given area
- Important for agriculture, forestry, and environmental monitoring
- Correlates with NDVI values

### Linear Regression
- Simple but effective model for predicting continuous values
- Assumes a linear relationship between input (NDVI) and output (biomass)
- Easy to interpret and understand

## Expected Results

- A trained linear regression model
- Clear visualizations showing NDVI vs biomass relationship
- Model accuracy metrics (R², RMSE, MAE)
- Understanding of how satellite data can predict ground conditions

## Next Steps

After completing this project, you can:
- Try different machine learning algorithms (Random Forest, Neural Networks)
- Use real satellite data from sources like Landsat or Sentinel
- Incorporate additional features (temperature, rainfall, soil type)
- Apply to specific regions or crop types

