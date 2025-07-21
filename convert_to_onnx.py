import tensorflow as tf
import tf2onnx
import onnx
import numpy as np
import os

def convert_h5_to_onnx(h5_path, onnx_path):
    """Convert H5 model to ONNX format"""
    print("=== CONVERTING H5 TO ONNX ===")
    
    try:
        # Load the H5 model
        print("Loading H5 model...")
        model = tf.keras.models.load_model(h5_path, compile=False)
        print(f"✓ Model loaded successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shapes: {[output.shape for output in model.outputs]}")
        
        # Convert to ONNX
        print("Converting to ONNX...")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=None,
            opset=13,
            output_path=onnx_path
        )
        
        print(f"✓ Model converted successfully to {onnx_path}")
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_onnx_model(onnx_path):
    """Test the ONNX model with sample data"""
    print("\n=== TESTING ONNX MODEL ===")
    
    try:
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(onnx_path)
        
        # Get input and output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"Input name: {input_info.name}")
        print(f"Input shape: {input_info.shape}")
        print(f"Input type: {input_info.type}")
        
        print(f"Number of outputs: {len(output_info)}")
        for i, output in enumerate(output_info):
            print(f"Output {i+1}: {output.name}, shape: {output.shape}")
        
        # Test with sample data
        sample_input = np.random.randn(1, 40, 216, 1).astype(np.float32)
        
        # Run inference
        result = session.run(None, {input_info.name: sample_input})
        
        print(f"✓ Inference successful")
        print(f"Category output shape: {result[0].shape}")
        print(f"Subcategory output shape: {result[1].shape}")
        print(f"Sample category prediction: {np.argmax(result[0])}")
        print(f"Sample subcategory prediction: {np.argmax(result[1])}")
        
        return True, input_info.name, [output.name for output in output_info]
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False, None, None

if __name__ == "__main__":
    h5_path = "audio_classifier.h5"
    onnx_path = "audio_classifier.onnx"
    
    if not os.path.exists(h5_path):
        print(f"❌ {h5_path} not found!")
    else:
        # Convert to ONNX
        if convert_h5_to_onnx(h5_path, onnx_path):
            # Test the ONNX model
            success, input_name, output_names = test_onnx_model(onnx_path)
            
            if success:
                print(f"\n🎉 SUCCESS!")
                print(f"✅ ONNX model ready: {onnx_path}")
                print(f"✅ Input name: {input_name}")
                print(f"✅ Output names: {output_names}")
                print(f"\nUse this information in your web application!")
