#!/usr/bin/env python3
"""
Setup script for Biomass Prediction Project

This script helps you get started with the project by:
1. Checking Python version
2. Installing required packages
3. Testing the installation
4. Running a quick demo
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    print(" Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("    Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"    Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_packages():
    """Check and install required packages."""
    print("\n Checking and installing required packages...")
    
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"    {package} is already installed")
        except ImportError:
            print(f"    Installing {package}...")
            if install_package(package):
                print(f"    {package} installed successfully")
            else:
                print(f"    Failed to install {package}")
                missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  Some packages could not be installed: {', '.join(missing_packages)}")
        print("   You may need to install them manually:")
        print("   pip install " + " ".join(missing_packages))
        return False
    
    print("    All required packages are available!")
    return True

def test_imports():
    """Test if all packages can be imported successfully."""
    print("\n Testing package imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import sklearn
        import seaborn as sns
        import scipy
        
        print("    All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"    Import error: {e}")
        return False

def run_quick_test():
    """Run a quick test to ensure everything works."""
    print("\n Running quick test...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create a simple plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Test Plot - Everything is working!')
        plt.xlabel('X')
        plt.ylabel('sin(X)')
        plt.grid(True, alpha=0.3)
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("    Quick test completed successfully!")
        print("   Test plot saved as 'test_plot.png'")
        return True
        
    except Exception as e:
        print(f"    Quick test failed: {e}")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n NEXT STEPS")
    print("=" * 40)
    print(" Setup complete! Here's what you can do next:")
    print("\n Quick Start:")
    print("   python demo.py                    # Run the quick demo")
    print("   python biomass_prediction.py      # Run the full project")
    
    print("\n Individual Components:")
    print("   python sample_data_generator.py   # Generate sample data")
    print("   python model_training.py          # Train the model")
    print("   python visualizations.py          # Create visualizations")
    
    print("\n Learn More:")
    print("   • Read README.md for detailed explanations")
    print("   • Open biomass_prediction_tutorial.ipynb in Jupyter")
    print("   • Modify the code to experiment with different approaches")
    
    print("\n Project Structure:")
    print("   • sample_data_generator.py  - Creates synthetic NDVI/biomass data")
    print("   • model_training.py         - Trains linear regression model")
    print("   • visualizations.py         - Creates comprehensive plots")
    print("   • biomass_prediction.py     - Main orchestration script")
    print("   • demo.py                   - Quick demonstration")
    
    print("\n Understanding the Project:")
    print("   • NDVI measures vegetation health from satellite imagery")
    print("   • Biomass is the total mass of living organisms")
    print("   • We use linear regression to predict biomass from NDVI")
    print("   • This has applications in agriculture, forestry, and ecology")

def main():
    """Main setup function."""
    print(" BIOMASS PREDICTION PROJECT - SETUP")
    print("=" * 50)
    print("This script will help you get started with the project.\n")
    
    # Check Python version
    if not check_python_version():
        print("\n Setup failed: Python version incompatible")
        return
    
    # Check and install packages
    if not check_and_install_packages():
        print("\n Setup failed: Could not install required packages")
        return
    
    # Test imports
    if not test_imports():
        print("\n Setup failed: Package imports failed")
        return
    
    # Run quick test
    if not run_quick_test():
        print("\n Setup failed: Quick test failed")
        return
    
    # Show next steps
    show_next_steps()
    
    print("\n Setup completed successfully!")
    print("You're ready to explore biomass prediction!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Setup interrupted by user.")
    except Exception as e:
        print(f"\n Unexpected error during setup: {e}")
        print("Please check the error message and try again.")
















