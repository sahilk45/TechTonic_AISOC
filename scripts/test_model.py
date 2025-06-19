import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def test_model_performance():
    """Test the trained model performance"""
    
    # Load data
    X = np.load("data/processed/features.npy")
    y = np.load("data/processed/labels.npy")
    
    # Load model
    model = tf.keras.models.load_model("models/keras_model.h5")
    
    # Prepare data
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    
    # Per-class confidence distribution
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        class_confidences = np.max(y_pred[class_mask], axis=1)
        plt.hist(class_confidences, alpha=0.7, label=class_name, bins=20)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/confidence_distribution.png')
    plt.show()

if __name__ == "__main__":
    test_model_performance()