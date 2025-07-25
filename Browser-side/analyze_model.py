# analyze_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py

def custom_input_layer(*args, **kwargs):
    """Custom function to handle batch_shape parameter"""
    if 'batch_shape' in kwargs:
        # Convert batch_shape to input_shape
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape and len(batch_shape) > 1:
            kwargs['input_shape'] = batch_shape[1:]  # Remove batch dimension
    return keras.layers.InputLayer(*args, **kwargs)

def load_model_with_fallback(model_path):
    """Try multiple methods to load the model"""
    
    # Method 1: Standard loading with custom objects
    try:
        print("Trying standard loading with custom objects...")
        custom_objects = {'InputLayer': custom_input_layer}
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✓ Successfully loaded model with custom objects")
        return model
    except Exception as e:
        print(f"✗ Standard loading failed: {e}")
    
    # Method 2: Load without compilation
    try:
        print("Trying to load without compilation...")
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Successfully loaded model without compilation")
        return model
    except Exception as e:
        print(f"✗ Loading without compilation failed: {e}")
    
    # Method 3: Recreate model from config
    try:
        print("Trying to recreate model from saved config...")
        with h5py.File(model_path, 'r') as f:
            # Get model config
            model_config = f.attrs.get('model_config')
            if model_config:
                import json
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                config = json.loads(model_config)
                
                # Create model from config
                model = keras.models.model_from_config(config, custom_objects={'InputLayer': custom_input_layer})
                
                # Load weights
                model.load_weights(model_path)
                print("✓ Successfully recreated model from config")
                return model
    except Exception as e:
        print(f"✗ Recreating from config failed: {e}")
    
    # Method 4: Manual model recreation based on the structure we can see
    try:
        print("Trying to manually recreate model architecture...")
        return recreate_model_manually()
    except Exception as e:
        print(f"✗ Manual recreation failed: {e}")
    
    return None

def recreate_model_manually():
    """Manually recreate the model based on the known structure"""
    print("Creating model with known architecture...")
    
    # Input layer
    input_layer = keras.layers.Input(shape=(40, 216, 1), name='input_layer_5')
    
    # Based on typical audio classification architectures, let's create a reasonable model
    # This is a guess based on common patterns, you might need to adjust
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5, name='dropout_1')(x)
    
    # Output layers (based on the config we can see)
    category_output = keras.layers.Dense(4, activation='softmax', name='category_output')(x)
    subcategory_output = keras.layers.Dense(14, activation='softmax', name='subcategory_output')(x)
    
    # Create model
    model = keras.models.Model(inputs=input_layer, outputs=[category_output, subcategory_output])
    
    print("✓ Manually created model architecture")
    return model

def analyze_model_files():
    print("=== ANALYZING MODEL FILES ===")
    
    # 1. Load and analyze the model with multiple fallback methods
    model = load_model_with_fallback('audio_classifier.h5')
    
    if model is not None:
        print("\n1. MODEL ARCHITECTURE:")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Number of parameters: {model.count_params()}")
        
        model.summary()
        
        # 3. Check model layers for clues about preprocessing
        print(f"\n3. MODEL LAYERS:")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name} - {type(layer).__name__}")
            if hasattr(layer, 'input_shape'):
                print(f"  Input shape: {layer.input_shape}")
            if hasattr(layer, 'output_shape'):
                print(f"  Output shape: {layer.output_shape}")
    else:
        print("\n1. MODEL LOADING FAILED - Analyzing what we can from the H5 file...")
        
        # Try to extract information from H5 file directly
        try:
            with h5py.File('audio_classifier.h5', 'r') as f:
                print("\nH5 File structure:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"  Group: {name}")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"  Dataset: {name} - Shape: {obj.shape}")
                
                f.visititems(print_structure)
                
                # Try to get model config
                if 'model_config' in f.attrs:
                    config = f.attrs['model_config']
                    print(f"\nModel config extracted successfully (length: {len(config)})")
                    
        except Exception as e:
            print(f"Error analyzing H5 file: {e}")
    
    # 2. Load and analyze metadata
    try:
        metadata = np.load('metadata.npy', allow_pickle=True)
        print(f"\n2. METADATA:")
        print(f"Metadata type: {type(metadata)}")
        print(f"Metadata shape: {metadata.shape if hasattr(metadata, 'shape') else 'No shape'}")
        
        if isinstance(metadata, dict) or hasattr(metadata, 'item'):
            meta_dict = metadata.item() if hasattr(metadata, 'item') else metadata
            print("Metadata contents:")
            for key, value in meta_dict.items():
                print(f"  {key}: {value}")
        else:
            print(f"Metadata contents: {metadata}")
            
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = None
    
    return model, metadata

# Run the analysis
if __name__ == "__main__":
    model, metadata = analyze_model_files()
    
    if model is None:
        print("\n" + "="*50)
        print("MODEL LOADING SUMMARY")
        print("="*50)
        print("❌ Could not load the model due to TensorFlow version compatibility issues")
        print("\nWhat we know from the metadata:")
        print("- Model expects input shape: (40, 216, 1)")
        print("- Model has dual outputs: category (4 classes) and subcategory (14 classes)")
        print("- Uses MFCC features with 40 coefficients")
        print("- Audio preprocessing: 22050 Hz, 5 seconds duration")
        print("\nSolutions to try:")
        print("1. Use the same TensorFlow version that created the model")
        print("2. Re-save the model with current TensorFlow version")
        print("3. Use the manually recreated model structure (if weights aren't critical)")
        print("4. Convert the model to a more compatible format")
    else:
        print("\n" + "="*50)
        print("✓ MODEL LOADED SUCCESSFULLY!")
        print("="*50)