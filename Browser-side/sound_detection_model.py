from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import os
import pickle
import tensorflow as tf
from typing import Dict, Any
import tempfile

app = FastAPI(title="Audio Classification API")

# Load your trained model and metadata
try:
    with open(r'D:\CA content\Python\Lang_Chain_Model\API\model1.pkl', 'rb') as f:
        model = pickle.load(f)

    metadata = np.load(r'D:\CA content\Python\Lang_Chain_Model\API\saved_model\metadata.npy', allow_pickle=True).item()
except Exception as e:
    raise RuntimeError(f"Failed to load model or metadata: {str(e)}")

# Parameters from your code
params = {
    'sample_rate': 22050,
    'duration': 5,  # seconds
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
}

def extract_mfcc(audio: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    """Extract MFCC features from audio."""
    try:
        # Ensure audio is the correct length
        target_length = sr * params['duration']
        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=params['n_mfcc'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length']
        )
        
        # Normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        return mfccs
    except Exception as e:
        raise ValueError(f"Error extracting MFCCs: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Classification API"}

@app.post("/predict", response_model=Dict[str, Any])
async def predict_audio(file: UploadFile = File(...)):
    """Endpoint to predict audio class from uploaded file using MFCCs."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        try:
            # Load audio file
            audio, sr = librosa.load(tmp_path, sr=params['sample_rate'])
            
            # Extract MFCC features
            mfccs = extract_mfcc(audio, sr, params)
            
            # Prepare input for model
            input_data = np.expand_dims(mfccs, axis=0)  # Add batch dimension
            # input_data = np.expand_dims(input_data, axis=-1)
            
            # Make prediction
            predictions = model.predict(input_data)
            
            # Process predictions based on your metadata
            predicted_category_idx = np.argmax(predictions[0])  
            predicted_category = metadata['category_mapping'][predicted_category_idx]
            
            # If have subcategories
            if 'subcategory_mapping' in metadata and len(predictions) > 1:
                predicted_subcategory_idx = np.argmax(predictions[1])
                predicted_subcategory = metadata['subcategory_mapping'][predicted_subcategory_idx]
            else:
                predicted_subcategory = None
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return {
                "category": predicted_category,
                "subcategory": predicted_subcategory,
                "confidence": float(np.max(predictions[0]))  # Confidence score
            }
            
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Endpoint to get information about the loaded model."""
    return {
        "input_shape": metadata.get('input_shape'),
        "categories": metadata.get('category_mapping'),
        "subcategories": metadata.get('subcategory_mapping', []),
        "parameters": params
    }