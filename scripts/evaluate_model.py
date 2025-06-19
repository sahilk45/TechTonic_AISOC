import numpy as np
import json
import keras
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_model_and_preprocessing():
    """Load model and CRITICAL preprocessing objects"""
    
    model_path = "models/keras_model.keras"
    scaler_path = "models/scaler.pkl"
    label_path = "models/label_encoder.json"
    
    missing_files = []
    for path, name in [(model_path, "model"), (scaler_path, "scaler"), (label_path, "labels")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure to run the improved training script first!")
        return None, None, None
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load the EXACT same scaler used in training
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {scaler_path}")
        
        # Load class names
        with open(label_path, "r") as f:
            class_names = json.load(f)
        print(f"✅ Class names loaded: {class_names}")
        
        return model, scaler, class_names
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def load_and_preprocess_data(scaler, class_names):
    """Load and preprocess data using the EXACT same method as training"""
    
    features_path = "data/processed/features.npy"
    labels_path = "data/processed/labels.npy"
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("❌ Data files not found. Run training script first!")
        return None, None
    
    try:
        X = np.load(features_path)
        y = np.load(labels_path)
        print(f"✅ Data loaded - Features: {X.shape}, Labels: {y.shape}")
        
        # Clean data (same as training)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Encode labels (same order as training)
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)  # Fit on class_names to ensure same order
        y_encoded = label_encoder.transform(y)
        
        # CRITICAL: Use the EXACT same scaler from training
        X_scaled = scaler.transform(X_clean)  # Use transform, NOT fit_transform
        
        print(f"✅ Data preprocessed:")
        print(f"   - Features: {X_scaled.shape}")
        print(f"   - Labels: {y_encoded.shape}")
        print(f"   - Unique labels: {np.unique(y_encoded)}")
        print(f"   - Feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        
        return X_scaled, y_encoded
        
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        return None, None

def detailed_model_analysis(model, X_test, y_test, class_names):
    """Perform detailed analysis of model performance"""
    
    print("\n🔍 Detailed Model Analysis:")
    
    # Check model weights
    try:
        first_layer = None
        for layer in model.layers:
            if hasattr(layer, 'weights') and len(layer.get_weights()) > 0:
                first_layer = layer
                break
        
        if first_layer:
            weights = first_layer.get_weights()[0]
            weight_std = np.std(weights)
            weight_mean = np.mean(np.abs(weights))
            
            print(f"   Weight statistics:")
            print(f"   - Standard deviation: {weight_std:.6f}")
            print(f"   - Mean absolute value: {weight_mean:.6f}")
            
            if weight_std > 0.01:
                print("   ✅ Model appears properly trained")
            else:
                print("   ⚠️  Weights may be undertrained")
        
    except Exception as e:
        print(f"   ⚠️  Could not analyze weights: {e}")
    
    # Prediction analysis
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Confidence analysis
    max_confidences = np.max(predictions, axis=1)
    avg_confidence = np.mean(max_confidences)
    min_confidence = np.min(max_confidences)
    max_confidence = np.max(max_confidences)
    
    print(f"   Prediction confidence:")
    print(f"   - Average: {avg_confidence:.3f}")
    print(f"   - Range: [{min_confidence:.3f}, {max_confidence:.3f}]")
    
    # Class distribution in predictions
    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    print(f"   Prediction distribution:")
    for pred_class, count in zip(unique_preds, pred_counts):
        percentage = count / len(y_pred) * 100
        print(f"   - {class_names[pred_class]}: {count} samples ({percentage:.1f}%)")
    
    # Check for bias toward one class
    most_frequent_percentage = max(pred_counts) / len(y_pred)
    if most_frequent_percentage > 0.5:
        most_freq_class = class_names[unique_preds[np.argmax(pred_counts)]]
        print(f"   ⚠️  Model heavily biased toward '{most_freq_class}' ({most_frequent_percentage:.1%})")
    
    return predictions, y_pred, avg_confidence

def analyze_per_class_performance(y_test, y_pred, class_names, cm):
    """Analyze performance for each class"""
    print("\n📈 Per-class Analysis:")
    
    # Calculate per-class metrics
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    class_precisions = cm.diagonal() / cm.sum(axis=0)
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    
    print(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Samples':<8}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        if i < len(class_accuracies):
            accuracy = class_accuracies[i] if not np.isnan(class_accuracies[i]) else 0
            precision = class_precisions[i] if not np.isnan(class_precisions[i]) else 0
            recall = class_recalls[i] if not np.isnan(class_recalls[i]) else 0
            samples = cm.sum(axis=1)[i]
            
            print(f"{class_name:<15} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {samples:<8}")
    
    # Identify best and worst performing classes
    valid_accuracies = class_accuracies[~np.isnan(class_accuracies)]
    if len(valid_accuracies) > 0:
        best_class_idx = np.nanargmax(class_accuracies)
        worst_class_idx = np.nanargmin(class_accuracies)
        
        print(f"\n🏆 Best performing class: {class_names[best_class_idx]} ({class_accuracies[best_class_idx]:.3f})")
        print(f"⚠️  Worst performing class: {class_names[worst_class_idx]} ({class_accuracies[worst_class_idx]:.3f})")

def create_detailed_visualizations(y_test, y_pred, predictions, class_names, accuracy):
    """Create comprehensive visualizations"""
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(15, 12))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Confidence Distribution
    plt.subplot(2, 2, 2)
    max_confidences = np.max(predictions, axis=1)
    plt.hist(max_confidences, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Maximum Confidence')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(max_confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_confidences):.3f}')
    plt.legend()
    
    # Per-class Accuracy
    plt.subplot(2, 2, 3)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    valid_indices = ~np.isnan(class_accuracies)
    valid_accuracies = class_accuracies[valid_indices]
    valid_classes = [class_names[i] for i in range(len(class_names)) if valid_indices[i]]
    
    bars = plt.bar(range(len(valid_accuracies)), valid_accuracies)
    plt.title('Per-class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(valid_accuracies)), valid_classes, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if valid_accuracies[i] > 0.8:
            bar.set_color('green')
        elif valid_accuracies[i] > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Class Distribution
    plt.subplot(2, 2, 4)
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    class_labels = [class_names[i] for i in unique_labels]
    plt.pie(label_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Test Set Class Distribution')
    
    plt.tight_layout()
    plt.savefig('models/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Separate confusion matrix for better visibility
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def save_detailed_report(accuracy, avg_confidence, y_test, y_pred, class_names, cm):
    """Save comprehensive evaluation report"""
    
    os.makedirs("models", exist_ok=True)
    
    with open("models/evaluation_report.txt", "w") as f:
        f.write("🚀 COMPREHENSIVE MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"📊 OVERALL PERFORMANCE:\n")
        f.write(f"   • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"   • Average Confidence: {avg_confidence:.3f}\n")
        f.write(f"   • Test Samples: {len(y_test)}\n")
        f.write(f"   • Number of Classes: {len(class_names)}\n\n")
        
        # Performance assessment
        if accuracy > 0.90:
            f.write("🎉 EXCEPTIONAL PERFORMANCE! Ready for production deployment!\n\n")
        elif accuracy > 0.85:
            f.write("🎉 EXCELLENT PERFORMANCE! Ready for TensorFlow.js conversion!\n\n")
        elif accuracy > 0.70:
            f.write("👍 GOOD PERFORMANCE! Consider fine-tuning for even better results.\n\n")
        elif accuracy > 0.50:
            f.write("⚠️  MODERATE PERFORMANCE. May need more training or data.\n\n")
        else:
            f.write("🚨 POOR PERFORMANCE! Requires significant improvement.\n\n")
        
        # Per-class analysis
        f.write("📈 PER-CLASS PERFORMANCE:\n")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_precisions = cm.diagonal() / cm.sum(axis=0)
        class_recalls = cm.diagonal() / cm.sum(axis=1)
        
        f.write(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Samples':<8}\n")
        f.write("-" * 60 + "\n")
        
        for i, class_name in enumerate(class_names):
            if i < len(class_accuracies):
                accuracy_val = class_accuracies[i] if not np.isnan(class_accuracies[i]) else 0
                precision_val = class_precisions[i] if not np.isnan(class_precisions[i]) else 0
                recall_val = class_recalls[i] if not np.isnan(class_recalls[i]) else 0
                samples = cm.sum(axis=1)[i]
                
                f.write(f"{class_name:<15} {accuracy_val:<10.3f} {precision_val:<10.3f} {recall_val:<10.3f} {samples:<8}\n")
        
        f.write("\n")
        
        # Classification report
        try:
            report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
            f.write("📋 DETAILED CLASSIFICATION REPORT:\n")
            f.write(report)
        except Exception as e:
            f.write(f"⚠️  Could not generate classification report: {e}\n")
        
        f.write("\n📁 Generated Files:\n")
        f.write("   • models/evaluation_report.txt (this file)\n")
        f.write("   • models/confusion_matrix.png\n")
        f.write("   • models/comprehensive_evaluation.png\n")

def evaluate_model_comprehensive():
    """Comprehensive model evaluation"""
    
    print("🚀 Starting Comprehensive Model Evaluation...")
    print("=" * 70)
    
    # Load everything
    print("🔍 Loading model and preprocessing objects...")
    model, scaler, class_names = load_model_and_preprocessing()
    
    if model is None:
        return
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    X_processed, y_encoded = load_and_preprocess_data(scaler, class_names)
    
    if X_processed is None:
        return
    
    # Split data with EXACT same parameters as training
    print("\n✂️  Splitting data (matching training exactly)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded,
        test_size=0.2,
        random_state=42,  # CRITICAL: Must match training
        stratify=y_encoded
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Detailed analysis
    predictions, y_pred, avg_confidence = detailed_model_analysis(model, X_test, y_test, class_names)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Performance assessment
    if accuracy > 0.90:
        print("🎉 EXCEPTIONAL PERFORMANCE! Ready for production deployment!")
    elif accuracy > 0.85:
        print("🎉 EXCELLENT PERFORMANCE! Ready for TensorFlow.js conversion!")
    elif accuracy > 0.70:
        print("👍 GOOD PERFORMANCE! Consider fine-tuning for even better results.")
    elif accuracy > 0.50:
        print("⚠️  MODERATE PERFORMANCE. May need more training or data.")
    else:
        print("🚨 POOR PERFORMANCE! Requires significant improvement.")
    
    # Create visualizations
    print("\n📊 Generating comprehensive visualizations...")
    cm = create_detailed_visualizations(y_test, y_pred, predictions, class_names, accuracy)
    
    # Per-class analysis
    analyze_per_class_performance(y_test, y_pred, class_names, cm)
    
    # Save detailed report
    print("\n💾 Saving comprehensive evaluation report...")
    save_detailed_report(accuracy, avg_confidence, y_test, y_pred, class_names, cm)
    
    # Print classification report to console
    print("\n📋 Detailed Classification Report:")
    try:
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        print(report)
    except Exception as e:
        print(f"⚠️  Could not generate classification report: {e}")
    
    print("\n✅ Evaluation completed successfully!")
    print("📁 Generated files:")
    print("   • models/evaluation_report.txt")
    print("   • models/confusion_matrix.png")
    print("   • models/comprehensive_evaluation.png")
    
    return accuracy, avg_confidence

if __name__ == "__main__":
    accuracy, confidence = evaluate_model_comprehensive()