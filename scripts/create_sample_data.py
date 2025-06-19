"""
This script creates sample processed data files for development/testing
when the full ESC-50 dataset is not available.
"""

import numpy as np
import json
from pathlib import Path
import os

def create_sample_processed_data():
    """Create sample processed data files for testing"""
    
    # Create directory structure
    processed_dir = Path("/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Define target classes (same as in your roadmap)
    target_classes = [
        'dog', 'siren', 'crying_baby', 'clock_alarm', 
        'glass_breaking', 'door_wood_knock', 'footsteps',
        'helicopter', 'chainsaw', 'cat'
    ]
    
    # Create sample data
    n_samples = 500  # 50 samples per class
    n_features = 86  # 40 MFCCs + 40 std + 6 other features
    
    print(f"Creating sample dataset with {n_samples} samples and {n_features} features...")
    
    # Generate synthetic features (in real scenario, these come from audio files)
    np.random.seed(42)  # For reproducibility
    
    # Create realistic-looking features
    features_list = []
    labels_list = []
    
    for class_idx, class_name in enumerate(target_classes):
        for sample_idx in range(n_samples // len(target_classes)):
            # Generate realistic MFCC-like features
            # MFCCs typically range from -50 to 50, with lower coefficients having higher values
            mfcc_features = np.random.normal(0, 10, 40)
            mfcc_features[0] = np.random.normal(20, 5)  # First coefficient usually higher
            
            # MFCC standard deviations
            mfcc_std = np.abs(np.random.normal(5, 2, 40))
            
            # Spectral features
            spectral_centroid = np.random.normal(2000, 500)  # Hz
            spectral_rolloff = np.random.normal(4000, 1000)  # Hz
            zero_crossing_rate = np.random.uniform(0.01, 0.3)
            
            # Additional features
            energy = np.random.uniform(0.001, 0.1)
            tempo = np.random.uniform(60, 180)
            chroma_mean = np.random.uniform(0, 1)
            
            # Combine all features
            features = np.concatenate([
                mfcc_features,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, zero_crossing_rate, 
                 energy, tempo, chroma_mean]
            ])
            
            features_list.append(features)
            labels_list.append(class_name)
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Save processed data
    np.save(processed_dir / "features.npy", X)
    np.save(processed_dir / "labels.npy", y)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Features saved to: {processed_dir / 'features.npy'}")
    print(f"Labels saved to: {processed_dir / 'labels.npy'}")
    
    # Create class mapping
    class_mapping = {cls: idx for idx, cls in enumerate(target_classes)}
    with open(processed_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"Class mapping saved to: {processed_dir / 'class_mapping.json'}")
    print(f"Class mapping: {class_mapping}")
    
    # Create feature statistics for normalization
    feature_stats = {
        "mean": X.mean(axis=0).tolist(),
        "std": X.std(axis=0).tolist(),
        "min": X.min(axis=0).tolist(),
        "max": X.max(axis=0).tolist()
    }
    
    with open(processed_dir / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)
    
    print(f"Feature statistics saved to: {processed_dir / 'feature_stats.json'}")
    
    # Create data info file
    data_info = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(target_classes),
        "classes": target_classes,
        "created_date": "2024-01-01",
        "sample_rate": 22050,
        "duration": 5.0,
        "feature_extraction": {
            "mfcc_coefficients": 40,
            "spectral_features": 3,
            "temporal_features": 3,
            "total_features": 86
        }
    }
    
    with open(processed_dir / "data_info.json", "w") as f:
        json.dump(data_info, f, indent=2)
    
    print(f"Data info saved to: {processed_dir / 'data_info.json'}")
    
    return X, y, class_mapping

def create_directory_structure():
    """Create the complete directory structure for the project"""
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/augmented",
        "notebooks",
        "scripts",
        "models",
        "../frontend/public/models",
        "../docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def verify_data_files():
    """Verify that all required data files exist"""
    
    required_files = [
        "../data/processed/features.npy",
        "../data/processed/labels.npy", 
        "../data/processed/class_mapping.json",
        "../data/processed/feature_stats.json",
        "../data/processed/data_info.json"
    ]
    
    print("\nVerifying data files:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - EXISTS")
        else:
            print(f"❌ {file_path} - MISSING")
    
    return all(Path(f).exists() for f in required_files)

if __name__ == "__main__":
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nCreating sample processed data...")
    X, y, class_mapping = create_sample_processed_data()
    
    print("\nVerifying files...")
    all_files_exist = verify_data_files()
    
    if all_files_exist:
        print("\n✅ All required data files created successfully!")
        print("\nNext steps:")
        print("1. Run the training script: python scripts/train_model.py")
        print("2. Convert model to TensorFlow.js: python scripts/convert_to_tfjs.py")
        print("3. Start the frontend: cd ../frontend && npm start")
    else:
        print("\n❌ Some files are missing. Please check the errors above.")