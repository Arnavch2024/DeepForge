"""
Carbon Impact Model Training Script
Trains ML models and saves them for later use
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare the carbon impact dataset with improved GPU-specific calculations"""
    df = pd.read_csv('synthetic_carbon_impact_dataset.csv')
    
    # Define realistic GPU power consumption and efficiency factors
    gpu_specs = {
        'CPU': {'power_watts': 65, 'efficiency_factor': 0.3, 'carbon_multiplier': 0.4},
        'RTX 3050': {'power_watts': 130, 'efficiency_factor': 0.6, 'carbon_multiplier': 0.7},
        'RTX 3060': {'power_watts': 170, 'efficiency_factor': 0.7, 'carbon_multiplier': 0.8},
        'RTX 4050': {'power_watts': 115, 'efficiency_factor': 0.75, 'carbon_multiplier': 0.65},
        'RTX 4090': {'power_watts': 450, 'efficiency_factor': 0.9, 'carbon_multiplier': 1.2},
        'A100': {'power_watts': 400, 'efficiency_factor': 0.95, 'carbon_multiplier': 1.0},
        'TPU': {'power_watts': 200, 'efficiency_factor': 1.2, 'carbon_multiplier': 0.5}
    }
    
    # Recalculate carbon emissions based on realistic GPU specifications
    def calculate_realistic_carbon(row):
        gpu_type = row['GPU_Type']
        training_hours = row['Training_Hours']
        model_params = row['Model_Parameters']
        dataset_size = row['Dataset_Size_MB']
        
        if gpu_type not in gpu_specs:
            return row['CO2_kg']  # Keep original if GPU not in specs
        
        specs = gpu_specs[gpu_type]
        
        # Base energy calculation: power * hours
        base_energy = (specs['power_watts'] / 1000) * training_hours
        
        # Model complexity factor (larger models need more energy)
        complexity_factor = 1 + (model_params / 1e9) * 0.5
        
        # Dataset size factor (larger datasets need more processing)
        dataset_factor = 1 + (dataset_size / 5000) * 0.3
        
        # Calculate total energy with efficiency
        total_energy = base_energy * complexity_factor * dataset_factor / specs['efficiency_factor']
        
        # Convert to CO2 (assuming 0.5 kg CO2 per kWh average grid emission)
        co2_kg = total_energy * 0.5 * specs['carbon_multiplier']
        
        return max(0.01, co2_kg)  # Minimum 0.01 kg CO2
    
    # Apply realistic carbon calculations
    print("🔄 Recalculating carbon emissions with realistic GPU specifications...")
    df['CO2_kg'] = df.apply(calculate_realistic_carbon, axis=1)
    df['Energy_kWh'] = df.apply(lambda row: (gpu_specs.get(row['GPU_Type'], {'power_watts': 200})['power_watts'] / 1000) * row['Training_Hours'], axis=1)
    
    # Encode GPU types
    le = LabelEncoder()
    df['GPU_Type_Encoded'] = le.fit_transform(df['GPU_Type'])
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"GPU Types: {df['GPU_Type'].unique()}")
    print("\nGPU Carbon Impact Summary:")
    gpu_summary = df.groupby('GPU_Type')['CO2_kg'].agg(['mean', 'min', 'max']).round(3)
    print(gpu_summary)
    
    return df, le

def train_models(df, le):
    """Train ML models to predict CO2 emissions with enhanced feature engineering"""
    # Prepare features and target with additional engineered features
    base_features = ['Model_Parameters', 'GPU_Type_Encoded', 'Training_Hours', 'Dataset_Size_MB', 'Energy_kWh']
    
    # Add engineered features for better GPU differentiation
    df['Params_per_Hour'] = df['Model_Parameters'] / (df['Training_Hours'] + 0.1)  # Avoid division by zero
    df['Energy_per_Param'] = df['Energy_kWh'] / (df['Model_Parameters'] / 1e6 + 0.1)
    df['GPU_Power_Factor'] = df['GPU_Type_Encoded'] * df['Energy_kWh']  # GPU-specific energy interaction
    
    features = base_features + ['Params_per_Hour', 'Energy_per_Param', 'GPU_Power_Factor']
    X = df[features]
    y = df['CO2_kg']
    
    print(f"📊 Feature correlation with CO2 emissions:")
    correlations = df[features + ['CO2_kg']].corr()['CO2_kg'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'CO2_kg':
            print(f"  {feature}: {corr:.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enhanced models with better hyperparameters
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'Enhanced Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(features, model.feature_importances_))
            print(f"  Top 3 important features:")
            for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {feat}: {imp:.3f}")
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred.tolist(),
            'actual': y_test.tolist()
        }
        
        # Save the model
        model_filename = f"{name.lower().replace(' ', '_')}_carbon_model.pkl"
        joblib.dump(model, model_filename)
        
        print(f"  MSE: {mse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Model saved as: {model_filename}")
    
    # Save the label encoder
    joblib.dump(le, 'gpu_label_encoder.pkl')
    print(f"\n💾 Label encoder saved as: gpu_label_encoder.pkl")
    
    # Save training metadata with enhanced features
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset_shape': df.shape,
        'gpu_types': df['GPU_Type'].unique().tolist(),
        'feature_names': features,
        'gpu_carbon_stats': df.groupby('GPU_Type')['CO2_kg'].agg(['mean', 'std']).round(4).to_dict(),
        'models': {
            name: {
                'mse': float(results[name]['mse']),
                'r2': float(results[name]['r2']),
                'filename': f"{name.lower().replace(' ', '_')}_carbon_model.pkl"
            }
            for name in results.keys()
        }
    }
    
    with open('training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Training metadata saved as: training_metadata.json")
    
    return results

def main():
    """Main training function"""
    print("🌱 Carbon Impact Model Training Starting...")
    
    # Load data
    df, le = load_and_prepare_data()
    
    # Train models
    print("\n🤖 Training ML models...")
    results = train_models(df, le)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_r2 = results[best_model_name]['r2']
    
    print(f"\n🏆 Best model: {best_model_name} (R² = {best_r2:.3f})")
    print("\n✅ Training complete! Models saved for visualization and API use.")

if __name__ == "__main__":
    main()