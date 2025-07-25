# test_onnx_model.py
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('audio_classifier.onnx')

print("=== TESTING ONNX MODEL ===")
for file in test_files:
    try:
        features = process_audio_python(file)  # Same preprocessing
        
        # Run ONNX inference
        results = session.run(None, {session.get_inputs()[0].name: features})
        
        category_pred = np.argmax(results[0])
        subcategory_pred = np.argmax(results[1])
        
        print(f"\n{file}:")
        print(f"Category: {category_pred} (confidence: {results[0][0][category_pred]:.3f})")
        print(f"Subcategory: {subcategory_pred} (confidence: {results[1][0][subcategory_pred]:.3f})")
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
