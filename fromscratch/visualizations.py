"""
Visualizations for Biomass Prediction Project

This script creates comprehensive visualizations to understand the data,
model performance, and make predictions with the trained model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_comprehensive_visualizations():
    """
    Create a comprehensive set of visualizations for the biomass prediction project.
    """
    print("🎨 Creating comprehensive visualizations for biomass prediction...")
    
    try:
        # Load the data
        df = pd.read_csv('sample_ndvi_biomass_data.csv')
        print(f"   Loaded {len(df)} data points")
    except FileNotFoundError:
        print("   Error: Could not find sample_ndvi_biomass_data.csv")
        print("   Please run sample_data_generator.py first!")
        return
    
    # Create the main visualization figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Data Overview - NDVI vs Biomass Scatter with Trend
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(df['ndvi'], df['biomass'], 
                          c=df['ndvi'], cmap='RdYlGn', alpha=0.7, s=30)
    ax1.set_xlabel('NDVI Value')
    ax1.set_ylabel('Biomass (tons/hectare)')
    ax1.set_title('NDVI vs Biomass Relationship\n(Colored by NDVI)')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['ndvi'], df['biomass'], 1)
    p = np.poly1d(z)
    ax1.plot(df['ndvi'], p(df['ndvi']), "r--", alpha=0.8, linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('NDVI Value')
    
    # 2. NDVI Distribution by Vegetation Type
    ax2 = plt.subplot(3, 3, 2)
    vegetation_types = ['Water/Barren', 'Sparse', 'Moderate', 'Dense']
    colors = ['blue', 'yellow', 'orange', 'green']
    
    for i, vtype in enumerate(vegetation_types):
        subset = df[df['vegetation_type'] == vtype]
        ax2.hist(subset['ndvi'], bins=20, alpha=0.7, 
                color=colors[i], label=vtype, edgecolor='black')
    
    ax2.set_xlabel('NDVI Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('NDVI Distribution by Vegetation Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Biomass Distribution by Vegetation Type
    ax3 = plt.subplot(3, 3, 3)
    for i, vtype in enumerate(vegetation_types):
        subset = df[df['vegetation_type'] == vtype]
        ax3.hist(subset['biomass'], bins=20, alpha=0.7, 
                color=colors[i], label=vtype, edgecolor='black')
    
    ax3.set_xlabel('Biomass (tons/hectare)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Biomass Distribution by Vegetation Type')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation Heatmap
    ax4 = plt.subplot(3, 3, 4)
    correlation_matrix = df[['ndvi', 'biomass']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax4, cbar_kws={'label': 'Correlation'})
    ax4.set_title('Correlation Matrix')
    
    # 5. Box Plot - Biomass by Vegetation Type
    ax5 = plt.subplot(3, 3, 5)
    df.boxplot(column='biomass', by='vegetation_type', ax=ax5)
    ax5.set_title('Biomass Distribution by Vegetation Type')
    ax5.set_xlabel('Vegetation Type')
    ax5.set_ylabel('Biomass (tons/hectare)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Violin Plot - NDVI by Vegetation Type
    ax6 = plt.subplot(3, 3, 6)
    sns.violinplot(data=df, x='vegetation_type', y='ndvi', ax=ax6)
    ax6.set_title('NDVI Distribution by Vegetation Type')
    ax6.set_xlabel('Vegetation Type')
    ax6.set_ylabel('NDVI Value')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Scatter with Vegetation Type Colors
    ax7 = plt.subplot(3, 3, 7)
    for i, vtype in enumerate(vegetation_types):
        subset = df[df['vegetation_type'] == vtype]
        ax7.scatter(subset['ndvi'], subset['biomass'], 
                   c=colors[i], label=vtype, alpha=0.7, s=30)
    
    ax7.set_xlabel('NDVI Value')
    ax7.set_ylabel('Biomass (tons/hectare)')
    ax7.set_title('NDVI vs Biomass by Vegetation Type')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Residual Analysis (if model exists)
    ax8 = plt.subplot(3, 3, 8)
    try:
        # Try to load and use the trained model
        model = LinearRegression()
        X = df['ndvi'].values.reshape(-1, 1)
        y = df['biomass'].values
        model.fit(X, y)
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        ax8.scatter(y_pred, residuals, alpha=0.6, color='purple')
        ax8.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax8.set_xlabel('Predicted Biomass')
        ax8.set_ylabel('Residuals')
        ax8.set_title('Residual Plot\n(Actual - Predicted)')
        ax8.grid(True, alpha=0.3)
        
    except Exception as e:
        ax8.text(0.5, 0.5, 'Run model_training.py\nto see residuals', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Residual Plot\n(Run model training first)')
    
    # 9. Prediction Examples
    ax9 = plt.subplot(3, 3, 9)
    try:
        # Show some prediction examples
        sample_ndvi = np.array([-0.5, 0, 0.3, 0.6, 0.9]).reshape(-1, 1)
        sample_predictions = model.predict(sample_ndvi)
        
        ax9.scatter(sample_ndvi, sample_predictions, 
                   c='red', s=100, marker='s', label='Predictions')
        ax9.scatter(df['ndvi'], df['biomass'], 
                   c='blue', alpha=0.3, s=20, label='Training Data')
        
        # Add prediction line
        ndvi_range = np.linspace(-0.5, 1, 100).reshape(-1, 1)
        biomass_range = model.predict(ndvi_range)
        ax9.plot(ndvi_range, biomass_range, 'r--', linewidth=2, label='Model')
        
        ax9.set_xlabel('NDVI Value')
        ax9.set_ylabel('Biomass (tons/hectare)')
        ax9.set_title('Model Predictions\n(Red squares = examples)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
    except Exception as e:
        ax9.text(0.5, 0.5, 'Run model_training.py\nto see predictions', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.set_title('Model Predictions\n(Run model training first)')
    
    plt.tight_layout()
    plt.savefig('comprehensive_visualizations.png', dpi=300, bbox_inches='tight')
    print("   Comprehensive visualizations saved as 'comprehensive_visualizations.png'")
    plt.show()

def create_prediction_examples():
    """
    Create visualizations showing how the model makes predictions.
    """
    print("🔮 Creating prediction examples...")
    
    try:
        # Load the data
        df = pd.read_csv('sample_ndvi_biomass_data.csv')
        
        # Train a simple model for demonstration
        model = LinearRegression()
        X = df['ndvi'].values.reshape(-1, 1)
        y = df['biomass'].values
        model.fit(X, y)
        
        print(f"   Model trained: Biomass = {model.coef_[0]:.2f} × NDVI + {model.intercept_:.2f}")
        
    except FileNotFoundError:
        print("   Error: Could not find sample_ndvi_biomass_data.csv")
        return
    
    # Create prediction examples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Example 1: Different NDVI values
    example_ndvi = np.array([-0.5, 0, 0.3, 0.6, 0.9])
    example_biomass = model.predict(example_ndvi.reshape(-1, 1))
    
    # Create a table-like visualization
    ax1.axis('tight')
    ax1.axis('off')
    
    table_data = []
    for i, (ndvi, biomass) in enumerate(zip(example_ndvi, example_biomass)):
        vegetation_type = "Water/Barren" if ndvi < 0 else \
                         "Sparse" if ndvi < 0.3 else \
                         "Moderate" if ndvi < 0.6 else "Dense"
        
        table_data.append([f"{ndvi:.1f}", f"{biomass:.1f}", vegetation_type])
    
    table = ax1.table(cellText=table_data,
                      colLabels=['NDVI Value', 'Predicted Biomass\n(tons/hectare)', 'Vegetation Type'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#E8F5E8')
    
    ax1.set_title('Biomass Prediction Examples\n(What the model predicts for different NDVI values)', 
                  fontsize=14, pad=20)
    
    # Example 2: Interactive prediction line
    ndvi_range = np.linspace(-0.5, 1, 100)
    biomass_range = model.predict(ndvi_range.reshape(-1, 1))
    
    ax2.scatter(df['ndvi'], df['biomass'], alpha=0.3, color='lightblue', s=20, label='Training Data')
    ax2.plot(ndvi_range, biomass_range, 'r-', linewidth=3, label='Model Prediction')
    
    # Highlight some example points
    ax2.scatter(example_ndvi, example_biomass, c='red', s=100, marker='s', 
               label='Example Predictions', zorder=5)
    
    # Add annotations
    for i, (ndvi, biomass) in enumerate(zip(example_ndvi, example_biomass)):
        ax2.annotate(f'NDVI={ndvi:.1f}\nBiomass={biomass:.1f}', 
                     xy=(ndvi, biomass), xytext=(10, 10),
                     textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_xlabel('NDVI Value')
    ax2.set_ylabel('Biomass (tons/hectare)')
    ax2.set_title('Model Prediction Line\n(Red squares show example predictions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
    print("   Prediction examples saved as 'prediction_examples.png'")
    plt.show()

def create_data_quality_analysis():
    """
    Create visualizations to analyze data quality and characteristics.
    """
    print("🔍 Creating data quality analysis...")
    
    try:
        df = pd.read_csv('sample_ndvi_biomass_data.csv')
    except FileNotFoundError:
        print("   Error: Could not find sample_ndvi_biomass_data.csv")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Analysis', fontsize=16)
    
    # 1. Data completeness and basic stats
    ax1 = axes[0, 0]
    stats_data = [
        ['Total Samples', len(df)],
        ['NDVI Range', f"{df['ndvi'].min():.3f} to {df['ndvi'].max():.3f}"],
        ['Biomass Range', f"{df['biomass'].min():.1f} to {df['biomass'].max():.1f}"],
        ['NDVI Mean', f"{df['ndvi'].mean():.3f}"],
        ['Biomass Mean', f"{df['biomass'].mean():.1f}"],
        ['Correlation', f"{df['ndvi'].corr(df['biomass']):.3f}"]
    ]
    
    ax1.axis('tight')
    ax1.axis('off')
    table = ax1.table(cellText=stats_data,
                      colLabels=['Metric', 'Value'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for i in range(len(stats_data) + 1):
        for j in range(2):
            if i == 0:
                table[(i, j)].set_facecolor('#2196F3')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#E3F2FD')
    
    ax1.set_title('Dataset Statistics', pad=20)
    
    # 2. Outlier detection using IQR method
    ax2 = axes[0, 1]
    
    # Calculate IQR for both variables
    ndvi_q1, ndvi_q3 = df['ndvi'].quantile([0.25, 0.75])
    ndvi_iqr = ndvi_q3 - ndvi_q1
    ndvi_outliers = df[(df['ndvi'] < ndvi_q1 - 1.5*ndvi_iqr) | 
                       (df['ndvi'] > ndvi_q3 + 1.5*ndvi_iqr)]
    
    biomass_q1, biomass_q3 = df['biomass'].quantile([0.25, 0.75])
    biomass_iqr = biomass_q3 - biomass_q1
    biomass_outliers = df[(df['biomass'] < biomass_q1 - 1.5*biomass_iqr) | 
                          (df['biomass'] > biomass_q3 + 1.5*biomass_iqr)]
    
    # Plot outliers
    ax2.scatter(df['ndvi'], df['biomass'], alpha=0.6, color='blue', s=20, label='Normal Data')
    ax2.scatter(ndvi_outliers['ndvi'], ndvi_outliers['biomass'], 
               color='red', s=50, marker='x', label='NDVI Outliers', zorder=5)
    ax2.scatter(biomass_outliers['ndvi'], biomass_outliers['biomass'], 
               color='orange', s=50, marker='s', label='Biomass Outliers', zorder=5)
    
    ax2.set_xlabel('NDVI Value')
    ax2.set_ylabel('Biomass (tons/hectare)')
    ax2.set_title(f'Outlier Detection\n(Red X: NDVI outliers, Orange Square: Biomass outliers)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Data distribution normality check
    ax3 = axes[1, 0]
    
    # Create Q-Q plots
    from scipy import stats
    
    # NDVI Q-Q plot
    stats.probplot(df['ndvi'], dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot: NDVI vs Normal Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Relationship strength visualization
    ax4 = axes[1, 1]
    
    # Create a hexbin plot to show density
    hb = ax4.hexbin(df['ndvi'], df['biomass'], gridsize=20, cmap='Blues')
    ax4.set_xlabel('NDVI Value')
    ax4.set_ylabel('Biomass (tons/hectare)')
    ax4.set_title('Data Density Heatmap\n(Darker = more data points)')
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax4)
    cb.set_label('Count')
    
    plt.tight_layout()
    plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
    print("   Data quality analysis saved as 'data_quality_analysis.png'")
    plt.show()

def main():
    """
    Main function to run all visualization components.
    """
    print("🎨 Starting Visualization Suite for Biomass Prediction Project")
    print("=" * 60)
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations()
    
    # Create prediction examples
    create_prediction_examples()
    
    # Create data quality analysis
    create_data_quality_analysis()
    
    print("\n All visualizations complete!")
    print("Generated files:")
    print("  • comprehensive_visualizations.png")
    print("  • prediction_examples.png")
    print("  • data_quality_analysis.png")
    print("\nThese visualizations help you understand:")
    print("  • The relationship between NDVI and biomass")
    print("  • Data quality and characteristics")
    print("  • How the model makes predictions")
    print("  • Potential areas for improvement")

if __name__ == "__main__":
    main()

