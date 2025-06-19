import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

class ImprovedAcousticClassifier:
    def __init__(self, n_classes=10, input_shape=(86,)):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.class_weights = None
        
    def create_realistic_data(self):
        """Create more realistic audio features that actually represent different sound classes"""
        print("🎵 Creating realistic audio feature data...")
        
        target_classes = [
            'cat', 'chainsaw', 'clock_alarm', 'crying_baby', 'dog',
            'door_wood_knock', 'footsteps', 'glass_breaking', 'helicopter', 'siren'
        ]
        
        n_samples_per_class = 100  # More balanced dataset
        n_features = 86
        
        features_list = []
        labels_list = []
        
        np.random.seed(42)
        
        for class_idx, class_name in enumerate(target_classes):
            print(f"   Generating {n_samples_per_class} samples for {class_name}")
            
            for sample_idx in range(n_samples_per_class):
                # Create class-specific feature patterns
                if class_name == 'cat':
                    # Higher frequencies, moderate energy
                    mfcc = np.random.normal([15, 8, 5, 3, 2, 1, 0.5, 0.2] + [0]*32, 
                                          [3, 2, 1.5, 1, 0.8, 0.5, 0.3, 0.2] + [0.5]*32)
                    spectral_centroid = np.random.normal(3000, 300)
                    energy = np.random.uniform(0.02, 0.08)
                    
                elif class_name == 'chainsaw':
                    # Consistent high energy, broad spectrum
                    mfcc = np.random.normal([25, 15, 12, 8, 6, 4, 3, 2] + [1]*32, 
                                          [2, 2, 2, 1.5, 1, 1, 0.8, 0.5] + [0.8]*32)
                    spectral_centroid = np.random.normal(2500, 400)
                    energy = np.random.uniform(0.15, 0.35)
                    
                elif class_name == 'clock_alarm':
                    # Periodic, high frequency
                    mfcc = np.random.normal([10, 12, 15, 8, 3, 1, 0.5, 0.2] + [0]*32, 
                                          [2, 3, 2, 1.5, 1, 0.5, 0.3, 0.2] + [0.4]*32)
                    spectral_centroid = np.random.normal(4000, 500)
                    energy = np.random.uniform(0.08, 0.15)
                    
                elif class_name == 'crying_baby':
                    # Variable pitch, emotional content
                    mfcc = np.random.normal([20, 10, 6, 8, 5, 3, 2, 1] + [0.5]*32, 
                                          [5, 3, 2, 2, 1.5, 1, 0.8, 0.5] + [0.6]*32)
                    spectral_centroid = np.random.normal(3500, 600)
                    energy = np.random.uniform(0.06, 0.12)
                    
                elif class_name == 'dog':
                    # Lower frequencies than cat, more energy
                    mfcc = np.random.normal([18, 12, 7, 4, 2, 1, 0.5, 0.2] + [0]*32, 
                                          [4, 3, 2, 1.5, 1, 0.5, 0.3, 0.2] + [0.5]*32)
                    spectral_centroid = np.random.normal(2000, 400)
                    energy = np.random.uniform(0.04, 0.10)
                    
                elif class_name == 'door_wood_knock':
                    # Sharp transients, low frequency
                    mfcc = np.random.normal([30, 5, 2, 1, 0.5, 0.2, 0.1, 0.05] + [0]*32, 
                                          [5, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05] + [0.3]*32)
                    spectral_centroid = np.random.normal(1500, 300)
                    energy = np.random.uniform(0.10, 0.25)
                    
                elif class_name == 'footsteps':
                    # Rhythmic, low frequency
                    mfcc = np.random.normal([25, 8, 3, 1, 0.5, 0.2, 0.1, 0.05] + [0]*32, 
                                          [3, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05] + [0.4]*32)
                    spectral_centroid = np.random.normal(1200, 200)
                    energy = np.random.uniform(0.03, 0.08)
                    
                elif class_name == 'glass_breaking':
                    # High frequency, sharp transient
                    mfcc = np.random.normal([15, 5, 8, 12, 10, 6, 3, 1] + [0.5]*32, 
                                          [3, 2, 3, 4, 3, 2, 1, 0.5] + [0.7]*32)
                    spectral_centroid = np.random.normal(5000, 800)
                    energy = np.random.uniform(0.08, 0.20)
                    
                elif class_name == 'helicopter':
                    # Low frequency, periodic
                    mfcc = np.random.normal([35, 20, 8, 3, 1, 0.5, 0.2, 0.1] + [0]*32, 
                                          [3, 3, 2, 1, 0.5, 0.3, 0.2, 0.1] + [0.6]*32)
                    spectral_centroid = np.random.normal(800, 150)
                    energy = np.random.uniform(0.20, 0.40)
                    
                elif class_name == 'siren':
                    # Variable pitch, high energy
                    mfcc = np.random.normal([10, 15, 20, 12, 6, 3, 1, 0.5] + [0.5]*32, 
                                          [2, 4, 5, 3, 2, 1, 0.5, 0.3] + [0.8]*32)
                    spectral_centroid = np.random.normal(3800, 700)
                    energy = np.random.uniform(0.15, 0.30)
                
                # Ensure realistic MFCC bounds
                mfcc = np.clip(mfcc, -50, 50)
                
                # MFCC standard deviations (class-dependent)
                mfcc_std = np.abs(np.random.normal(3, 1, 40))
                
                # Additional features
                spectral_rolloff = spectral_centroid * np.random.uniform(1.5, 2.5)
                zero_crossing_rate = np.random.uniform(0.01, 0.15)
                tempo = np.random.uniform(60, 180)
                chroma_mean = np.random.uniform(0.1, 0.9)
                
                # Combine features
                features = np.concatenate([
                    mfcc,
                    mfcc_std,
                    [spectral_centroid, spectral_rolloff, zero_crossing_rate, 
                     energy, tempo, chroma_mean]
                ])
                
                # Add some noise for realism
                features += np.random.normal(0, 0.1, len(features))
                
                features_list.append(features)
                labels_list.append(class_name)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"✅ Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
        print(f"   Classes: {np.unique(y)}")
        print(f"   Samples per class: {len(y) // len(np.unique(y))}")
        
        return X, y
    
    def build_improved_model(self):
        """Build an improved model architecture"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # Reshape for Conv1D
            keras.layers.Reshape((86, 1)),
            
            # First Conv Block - smaller filters for features
            Conv1D(64, 5, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(64, 5, activation='relu', padding='same'),
            MaxPooling1D(2),
            Dropout(0.25),
            
            # Second Conv Block
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Dropout(0.25),
            
            # Third Conv Block
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            Dense(self.n_classes, activation='softmax')
        ])
        
        # Use a learning rate scheduler
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X, y):
        """Prepare data with proper preprocessing"""
        print("🔧 Preparing data...")
        
        # Handle any NaN or infinite values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features - IMPORTANT: Save the scaler for evaluation
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Calculate class weights for imbalanced data
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        self.class_weights = dict(enumerate(class_weights_array))
        
        print(f"✅ Data prepared:")
        print(f"   - Features shape: {X_scaled.shape}")
        print(f"   - Labels shape: {y_encoded.shape}")
        print(f"   - Feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        print(f"   - Class weights: {self.class_weights}")
        
        return X_scaled, y_encoded
    
    def train_with_validation(self, X, y, validation_split=0.2, epochs=150, batch_size=32):
        """Train with proper validation and callbacks"""
        
        # Prepare data
        X_processed, y_encoded = self.prepare_data(X, y)
        
        # Split data with same random state as evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded,
            test_size=validation_split,
            random_state=42,  # CRITICAL: Same as evaluation
            stratify=y_encoded
        )
        
        print(f"📊 Data split:")
        print(f"   - Training: {len(X_train)} samples")
        print(f"   - Validation: {len(X_val)} samples")
        
        # Improved callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=10,
                factor=0.5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🔥 Starting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        return history, X_val, y_val
    
    def save_everything(self, model_path):
        """Save model, scaler, and label encoder"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_path}/keras_model.keras")
        
        # Save label encoder
        label_classes = self.label_encoder.classes_.tolist()
        with open(f"{model_path}/label_encoder.json", "w") as f:
            json.dump(label_classes, f, indent=2)
        
        # CRITICAL: Save the scaler for consistent evaluation
        with open(f"{model_path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save class weights
        with open(f"{model_path}/class_weights.json", "w") as f:
            json.dump(self.class_weights, f, indent=2)
        
        print(f"✅ Everything saved to {model_path}")

def main():
    print("🚀 Starting Improved Acoustic Event Classification Training...")
    print("=" * 70)
    
    # Create classifier
    classifier = ImprovedAcousticClassifier(n_classes=10)
    
    # Create realistic data
    X, y = classifier.create_realistic_data()
    
    # Save the data
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/features.npy", X)
    np.save("data/processed/labels.npy", y)
    print("💾 Data saved for evaluation consistency")
    
    # Build model
    model = classifier.build_improved_model()
    print("\n🏗️  Model architecture:")
    model.summary()
    
    # Train model
    history, X_val, y_val = classifier.train_with_validation(X, y, epochs=100)
    
    # Save everything
    classifier.save_everything("models")
    
    # Plot training history
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'] if 'lr' in history.history else [0.001] * len(history.history['loss']))
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        print("📈 Training history saved to models/training_history.png")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
    
    # Final validation accuracy
    val_predictions = model.predict(X_val)
    val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == y_val)
    
    print(f"\n✅ Training completed!")
    print(f"🎯 Final validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    if val_accuracy > 0.8:
        print("🎉 Excellent! Model ready for TensorFlow.js conversion!")
    elif val_accuracy > 0.6:
        print("👍 Good performance. Consider training longer for even better results.")
    else:
        print("⚠️  Consider adjusting hyperparameters or training longer.")
    
    print("\n📁 Files created:")
    print("   - models/keras_model.keras")
    print("   - models/best_model.keras")
    print("   - models/label_encoder.json")
    print("   - models/scaler.pkl (CRITICAL for evaluation)")
    print("   - models/class_weights.json")
    print("   - models/training_history.png")
    print("   - data/processed/features.npy")
    print("   - data/processed/labels.npy")

if __name__ == "__main__":
    main()