"""
Sample Data Generator for Biomass Prediction Project

This script generates synthetic NDVI and biomass data to simulate real-world conditions.
We'll create realistic relationships between vegetation health (NDVI) and biomass.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def generate_ndvi_data(n_samples=1000, noise_level=0.1):
    """
    Generate synthetic NDVI data that mimics satellite imagery.
    
    NDVI values typically range from -1 to 1:
    - -1 to 0: Water, snow, or barren land
    - 0 to 0.3: Sparse vegetation
    - 0.3 to 0.6: Moderate vegetation
    - 0.6 to 1: Dense, healthy vegetation
    
    Args:
        n_samples (int): Number of data points to generate
        noise_level (float): Amount of random noise to add
        
    Returns:
        numpy.ndarray: Array of NDVI values
    """
    print(" Generating synthetic NDVI data...")
    
    # Create a realistic distribution of NDVI values
    # Most areas have moderate vegetation, fewer have very high or very low values
    
    # Generate base NDVI values with a realistic distribution
    # Using a beta distribution to create values between 0 and 1
    base_ndvi = np.random.beta(2, 1.5, n_samples)
    
    # Add some negative values for water/barren areas (about 10% of data)
    water_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    base_ndvi[water_indices] = np.random.uniform(-0.5, 0, size=len(water_indices))
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level, n_samples)
    ndvi_values = np.clip(base_ndvi + noise, -1, 1)
    
    print(f"   Generated {n_samples} NDVI values")
    print(f"   NDVI range: {ndvi_values.min():.3f} to {ndvi_values.max():.3f}")
    print(f"   Mean NDVI: {ndvi_values.mean():.3f}")
    
    return ndvi_values

def generate_biomass_data(ndvi_values, base_biomass=50, max_biomass=200, noise_level=0.15):
    """
    Generate biomass data that correlates with NDVI values.
    
    Biomass typically increases with NDVI, but the relationship isn't perfectly linear.
    We'll add some realistic non-linearity and noise.
    
    Args:
        ndvi_values (numpy.ndarray): Array of NDVI values
        base_biomass (float): Base biomass value
        max_biomass (float): Maximum biomass value
        noise_level (float): Amount of random noise to add
        
    Returns:
        numpy.ndarray: Array of biomass values (tons/hectare)
    """
    print(" Generating synthetic biomass data...")
    
    # Create a realistic relationship between NDVI and biomass
    # Biomass increases with NDVI, but with diminishing returns
    
    # Convert NDVI to 0-1 scale for easier biomass calculation
    ndvi_normalized = (ndvi_values + 1) / 2  # Convert from [-1,1] to [0,1]
    
    # Create non-linear relationship: biomass = base + (max-base) * (ndvi^1.5)
    # This gives diminishing returns at high NDVI values
    biomass_values = base_biomass + (max_biomass - base_biomass) * (ndvi_normalized ** 1.5)
    
    # Add realistic noise (proportional to biomass value)
    proportional_noise = np.random.normal(0, noise_level, len(biomass_values))
    biomass_values = biomass_values * (1 + proportional_noise)
    
    # Ensure biomass values are positive
    biomass_values = np.maximum(biomass_values, 0)
    
    print(f"   Generated {len(biomass_values)} biomass values")
    print(f"   Biomass range: {biomass_values.min():.1f} to {biomass_values.max():.1f} tons/hectare")
    print(f"   Mean biomass: {biomass_values.mean():.1f} tons/hectare")
    
    return biomass_values

def create_sample_dataset(n_samples=1000, save_to_file=True):
    """
    Create a complete dataset with NDVI and biomass values.
    
    Args:
        n_samples (int): Number of data points to generate
        save_to_file (bool): Whether to save the dataset to CSV
        
    Returns:
        pandas.DataFrame: Dataset with NDVI and biomass columns
    """
    print(" Creating complete sample dataset...")
    
    # Generate NDVI data
    ndvi_values = generate_ndvi_data(n_samples)
    
    # Generate corresponding biomass data
    biomass_values = generate_biomass_data(ndvi_values)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ndvi': ndvi_values,
        'biomass': biomass_values
    })
    
    # Add some metadata columns for educational purposes
    df['vegetation_type'] = pd.cut(ndvi_values, 
                                   bins=[-1, 0, 0.3, 0.6, 1], 
                                   labels=['Water/Barren', 'Sparse', 'Moderate', 'Dense'])
    
    # Calculate correlation
    correlation = df['ndvi'].corr(df['biomass'])
    print(f"   Correlation between NDVI and biomass: {correlation:.3f}")
    
    if save_to_file:
        filename = 'sample_ndvi_biomass_data.csv'
        df.to_csv(filename, index=False)
        print(f"   Dataset saved to {filename}")
    
    return df

def visualize_data_distribution(df):
    """
    Create simple visualizations to show the generated data.
    
    Args:
        df (pandas.DataFrame): Dataset with NDVI and biomass columns
    """
    print(" Creating data distribution visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Sample Data Distribution', fontsize=16)
    
    # NDVI histogram
    axes[0, 0].hist(df['ndvi'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].set_title('NDVI Distribution')
    axes[0, 0].set_xlabel('NDVI Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['ndvi'].mean(), color='red', linestyle='--', label=f'Mean: {df["ndvi"].mean():.3f}')
    axes[0, 0].legend()
    
    # Biomass histogram
    axes[0, 1].hist(df['biomass'], bins=30, alpha=0.7, color='brown', edgecolor='black')
    axes[0, 1].set_title('Biomass Distribution')
    axes[0, 1].set_xlabel('Biomass (tons/hectare)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['biomass'].mean(), color='red', linestyle='--', label=f'Mean: {df["biomass"].mean():.1f}')
    axes[0, 1].legend()
    
    # Scatter plot
    axes[1, 0].scatter(df['ndvi'], df['biomass'], alpha=0.6, color='darkgreen')
    axes[1, 0].set_title('NDVI vs Biomass Relationship')
    axes[1, 0].set_xlabel('NDVI Value')
    axes[1, 0].set_ylabel('Biomass (tons/hectare)')
    
    # Vegetation type distribution
    vegetation_counts = df['vegetation_type'].value_counts()
    axes[1, 1].bar(vegetation_counts.index, vegetation_counts.values, color=['blue', 'yellow', 'orange', 'green'])
    axes[1, 1].set_title('Vegetation Type Distribution')
    axes[1, 1].set_xlabel('Vegetation Type')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'data_distribution.png'")
    plt.show()

if __name__ == "__main__":
    print(" Starting Sample Data Generation for Biomass Prediction Project")
    print("=" * 60)
    
    # Create the dataset
    df = create_sample_dataset(n_samples=1000)
    
    # Show basic statistics
    print("\n Dataset Summary:")
    print(df.describe())
    
    print("\n Vegetation Type Breakdown:")
    print(df['vegetation_type'].value_counts())
    
    # Create visualizations
    print("\n Creating visualizations...")
    visualize_data_distribution(df)
    
    print("\n Sample data generation complete!")
    print("You can now use this data to train your biomass prediction model.")

