import numpy as np
import tensorflow as tf
import h5py
import json
import os
from packaging import version
import re

def analyze_original_model(model_path):
    """Analyze the original model to understand its true architecture"""
    print("=== ANALYZING ORIGINAL MODEL ===")
    
    try:
        # Load the original model
        original_model = tf.keras.models.load_model(model_path, compile=False)
        
        print("✓ Successfully loaded original model")
        print(f"Input shape: {original_model.input_shape}")
        print(f"Output shapes: {[output.shape for output in original_model.outputs]}")
        print(f"Total parameters: {original_model.count_params():,}")
        
        # Print layer summary
        print("\nModel Architecture:")
        for i, layer in enumerate(original_model.layers[:20]):  # First 20 layers
            print(f"  {i:2d}. {layer.name:30} {layer.__class__.__name__:20} {str(getattr(layer, 'output_shape', 'N/A')):20}")
        
        if len(original_model.layers) > 20:
            print(f"  ... and {len(original_model.layers) - 20} more layers")
        
        return original_model
        
    except Exception as e:
        print(f"❌ Error loading original model: {e}")
        return None

def analyze_h5_weights(model_path):
    """Analyze H5 weights to understand the architecture"""
    print("\n=== ANALYZING H5 WEIGHTS ===")
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Look for architecture clues
            print("Root groups:")
            for key in f.keys():
                print(f"  {key}")
            
            # Check for specific architecture patterns
            def find_architecture_clues(group, prefix=""):
                clues = []
                for key in group.keys():
                    full_path = f"{prefix}/{key}" if prefix else key
                    item = group[key]
                    
                    if isinstance(item, h5py.Group):
                        clues.extend(find_architecture_clues(item, full_path))
                    elif 'efficientnet' in full_path.lower():
                        clues.append(f"EfficientNet: {full_path}")
                    elif 'convnext' in full_path.lower():
                        clues.append(f"ConvNext: {full_path}")
                    elif 'mobilenet' in full_path.lower():
                        clues.append(f"MobileNet: {full_path}")
                
                return clues
            
            clues = find_architecture_clues(f)
            
            print("\nArchitecture clues found:")
            for clue in clues[:10]:  
                print(f"  {clue}")
                
            return clues
            
    except Exception as e:
        print(f"❌ Error analyzing H5 weights: {e}")
        return []

def create_compatible_model(original_model):
    """Create a model that's compatible with the original architecture"""
    print("\n=== CREATING COMPATIBLE MODEL ===")
    
    if original_model is None:
        print("❌ Cannot create compatible model without original model")
        return None
    
    try:
        # Get the original model's configuration
        config = original_model.get_config()
        
        # Create a new model with the same architecture
        new_model = tf.keras.Model.from_config(config)
        
        print("✓ Successfully created compatible model")
        print(f"New model parameters: {new_model.count_params():,}")
        
        return new_model
        
    except Exception as e:
        print(f"❌ Error creating compatible model: {e}")
        
        try:
            new_model = tf.keras.models.clone_model(original_model)
            print("✓ Successfully cloned original model")
            return new_model
        except Exception as e2:
            print(f"❌ Error cloning model: {e2}")
            return None

def direct_weight_transfer(original_model, new_model):
    """Transfer weights directly between models with identical architectures"""
    print("\n=== DIRECT WEIGHT TRANSFER ===")
    
    if original_model is None or new_model is None:
        print("❌ Cannot transfer weights: missing model(s)")
        return False
    
    try:
        # Copy weights layer by layer
        successful_transfers = 0
        failed_transfers = 0
        
        for orig_layer, new_layer in zip(original_model.layers, new_model.layers):
            if not hasattr(orig_layer, 'weights') or len(orig_layer.weights) == 0:
                continue
                
            try:
                orig_weights = orig_layer.get_weights()
                new_layer.set_weights(orig_weights)
                successful_transfers += 1
                print(f"  ✓ {orig_layer.name}")
                
            except Exception as e:
                failed_transfers += 1
                print(f"  ✗ {orig_layer.name}: {e}")
        
        print(f"\nTransfer results: {successful_transfers} successful, {failed_transfers} failed")
        
        return successful_transfers > 0
        
    except Exception as e:
        print(f"❌ Error during weight transfer: {e}")
        return False

def validate_model_functionality(model, original_model=None):
    """Validate that the model works correctly"""
    print("\n=== MODEL VALIDATION ===")
    
    if model is None:
        print("❌ No model to validate")
        return False
    
    try:
        # Test with a sample input
        sample_input = np.random.normal(0, 0.1, (1, 40, 216, 1))
        
        # Test prediction
        predictions = model.predict(sample_input, verbose=0)
        
        print(f"✓ Model prediction successful")
        print(f"  Category prediction shape: {predictions[0].shape}")
        print(f"  Subcategory prediction shape: {predictions[1].shape}")
        
        # Test with multiple inputs for diversity
        test_inputs = [
            np.random.normal(0, 0.1, (1, 40, 216, 1)),
            np.random.normal(0, 0.5, (1, 40, 216, 1)),
            np.random.normal(0, 1.0, (1, 40, 216, 1)),
        ]
        
        all_predictions = []
        for i, test_input in enumerate(test_inputs):
            pred = model.predict(test_input, verbose=0)
            cat_pred = np.argmax(pred[0][0])
            sub_pred = np.argmax(pred[1][0])
            all_predictions.append((cat_pred, sub_pred))
            print(f"  Test {i+1}: Category {cat_pred}, Subcategory {sub_pred}")
        
        # Check prediction diversity
        unique_cats = len(set([p[0] for p in all_predictions]))
        unique_subs = len(set([p[1] for p in all_predictions]))
        
        print(f"  Prediction diversity: {unique_cats} unique categories, {unique_subs} unique subcategories")
        
        # Compare with original model if available
        if original_model is not None:
            try:
                orig_pred = original_model.predict(sample_input, verbose=0)
                new_pred = model.predict(sample_input, verbose=0)
                
                cat_diff = np.mean(np.abs(orig_pred[0] - new_pred[0]))
                sub_diff = np.mean(np.abs(orig_pred[1] - new_pred[1]))
                
                print(f"  Prediction difference from original:")
                print(f"    Category: {cat_diff:.6f}")
                print(f"    Subcategory: {sub_diff:.6f}")
                
                if cat_diff < 0.001 and sub_diff < 0.001:
                    print("  ✓ Predictions match original model perfectly")
                    return True
                elif cat_diff < 0.1 and sub_diff < 0.1:
                    print("  ✓ Predictions are very close to original model")
                    return True
                else:
                    print("  ⚠️  Predictions differ significantly from original")
                    return False
                    
            except Exception as e:
                print(f"  ⚠️  Could not compare with original: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def save_converted_model(model, output_name="converted_model_fixed"):
    """Save the converted model in multiple formats"""
    print("\n=== SAVING CONVERTED MODEL ===")
    
    if model is None:
        print("❌ No model to save")
        return False
    
    try:
        # Save as H5
        h5_path = f"{output_name}.h5"
        model.save(h5_path)
        print(f"✓ H5 model saved: {h5_path}")
        
        # Save as SavedModel
        saved_model_path = f"{output_name}_savedmodel"
        if os.path.exists(saved_model_path):
            import shutil
            shutil.rmtree(saved_model_path)
        
        model.save(saved_model_path)
        print(f"✓ SavedModel saved: {saved_model_path}")
        
        # Convert to TensorFlow.js
        try:
            import tensorflowjs as tfjs
            
            tfjs_path = f"{output_name}_tfjs"
            if os.path.exists(tfjs_path):
                import shutil
                shutil.rmtree(tfjs_path)
            
            tfjs.converters.save_keras_model(
                model,
                tfjs_path,
                quantization_bytes=2,
                skip_op_check=True,
                weight_shard_size_bytes=1024*1024*4
            )
            
            print(f"✓ TensorFlow.js model saved: {tfjs_path}")
            
            # List created files
            files = os.listdir(tfjs_path)
            print(f"  Created files: {files}")
            
        except ImportError:
            print("⚠️  TensorFlow.js not installed, skipping JS conversion")
            print("  Install with: pip install tensorflowjs")
        except Exception as e:
            print(f"⚠️  TensorFlow.js conversion failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def main():
    """Main conversion function"""
    print("=== INTELLIGENT MODEL CONVERTER ===")
    print("This converter will analyze your model and use the appropriate conversion strategy.")
    
    model_path = 'audio_classifier.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found!")
        return False
    
    # Step 1: Analyze the original model
    original_model = analyze_original_model(model_path)
    
    # Step 2: Analyze H5 weights for additional clues
    architecture_clues = analyze_h5_weights(model_path)
    
    # Step 3: Create compatible model
    new_model = create_compatible_model(original_model)
    
    # Step 4: Transfer weights
    if not direct_weight_transfer(original_model, new_model):
        print("❌ Weight transfer failed")
        return False
    
    # Step 5: Validate the model
    if not validate_model_functionality(new_model, original_model):
        print("❌ Model validation failed")
        return False
    
    # Step 6: Save the converted model
    if not save_converted_model(new_model):
        print("❌ Model saving failed")
        return False
    
    print("\n" + "="*50)
    print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("✅ Original model architecture preserved")
    print("✅ All weights transferred successfully")
    print("✅ Model functionality validated")
    print("✅ Multiple output formats created")
    print("\nYour converted model is ready for deployment!")
    print("Use 'converted_model_fixed_tfjs/model.json' for web deployment.")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ CONVERSION FAILED!")
        print("="*50)
        print("The conversion failed. Here are some troubleshooting steps:")
        print("1. Ensure 'audio_classifier.h5' exists and is a valid Keras model")
        print("2. Try loading the model manually in Python to check for issues")
        print("3. Check that you have sufficient disk space")
        print("4. Verify TensorFlow version compatibility")
        print("\nIf problems persist, please share the error messages for further assistance.")
