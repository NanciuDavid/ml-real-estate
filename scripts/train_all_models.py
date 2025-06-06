#!/usr/bin/env python3
"""
Main script to train and compare all machine learning models
for real estate price prediction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """
    Main function to orchestrate the entire ML pipeline:
    1. Load and preprocess data
    2. Train multiple models
    3. Evaluate and compare results
    4. Generate reports
    """
    
    print("🏠 ML Real Estate Price Prediction Pipeline")
    print("=" * 50)
    
    # TODO: Implement data loading
    print("📊 Step 1: Loading and preprocessing data...")
    # from src.data.data_loader import load_data
    # from src.data.data_cleaner import clean_data
    # from src.data.feature_engineer import engineer_features
    
    # TODO: Implement model training
    print("🤖 Step 2: Training multiple ML models...")
    # from src.models.model_trainer import train_all_models
    
    # TODO: Implement evaluation
    print("📈 Step 3: Evaluating and comparing models...")
    # from src.evaluation.model_comparison import compare_models
    
    # TODO: Implement reporting
    print("📋 Step 4: Generating reports...")
    # from src.visualization.results_plots import generate_report
    
    print("✅ Pipeline completed successfully!")
    print("📁 Check results/ folder for outputs")

if __name__ == "__main__":
    main() 