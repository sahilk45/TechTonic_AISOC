"""Test script to check TensorFlow/Keras imports"""

import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        import keras
        print(f"Keras version: {keras.__version__}")
    except ImportError:
        print("Direct Keras import failed")
    
    # Test the problematic imports
    print("\nTesting imports:")
    
    try:
        from tensorflow.keras.models import Sequential
        print("✓ Sequential import successful")
    except ImportError as e:
        print(f"✗ Sequential import failed: {e}")
    
    try:
        from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
        print("✓ Layers import successful")
    except ImportError as e:
        print(f"✗ Layers import failed: {e}")
    
    try:
        from tensorflow.keras.optimizers import Adam
        print("✓ Adam optimizer import successful")
    except ImportError as e:
        print(f"✗ Adam optimizer import failed: {e}")
    
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        print("✓ Callbacks import successful")
    except ImportError as e:
        print(f"✗ Callbacks import failed: {e}")
    
    print("\n✓ All imports successful! Your code should work.")

except ImportError as e:
    print(f"TensorFlow import failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Check environment variables
import os
legacy_keras = os.environ.get('TF_USE_LEGACY_KERAS', 'Not set')
print(f"\nTF_USE_LEGACY_KERAS: {legacy_keras}")

print("\nIf imports failed, try one of these solutions:")
print("1. pip install tf_keras && set TF_USE_LEGACY_KERAS=1")
print("2. pip install tensorflow==2.13.0 keras==2.13.1")
print("3. Update your code to use direct 'import keras' instead of 'from tensorflow.keras'")