import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ESC50Preprocessor:
    def __init__(self, data_path, sample_rate=22050, duration=5):
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = sample_rate * duration
        
        # Target classes for our app
        self.target_classes = [
            'dog', 'siren', 'crying_baby', 'clock_alarm', 
            'glass_breaking', 'door_wood_knock', 'footsteps',
            'helicopter', 'chainsaw', 'cat'
        ]
    
    def load_audio_file(self, file_path):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), 'constant')
            else:
                audio = audio[:self.n_samples]
                
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract MFCC features from audio"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40)
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        features = np.vstack([mfccs, spectral_centroids, spectral_rolloff, zero_crossing_rate])
        
        features_mean = np.mean(features, axis=1)
        features_std = np.std(features, axis=1)
        
        return np.concatenate([features_mean, features_std])
    
    def process_dataset(self):
        """Process entire ESC-50 dataset"""
        metadata_path = self.data_path / "ESC-50-master/meta/esc50.csv"
        metadata = pd.read_csv(metadata_path)
        
        filtered_metadata = metadata[metadata['category'].isin(self.target_classes)]
        
        features_list = []
        labels_list = []
        
        print(f"Processing {len(filtered_metadata)} audio files...")
        
        for idx, row in filtered_metadata.iterrows():
            audio_path = self.data_path / "ESC-50-master/audio" / row['filename']
            
            # Load audio
            audio = self.load_audio_file(audio_path)
            if audio is None:
                continue
            
            # Extract features
            features = self.extract_features(audio)
            
            features_list.append(features)
            labels_list.append(row['category'])
            
            if len(features_list) % 50 == 0:
                print(f"Processed {len(features_list)} files...")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Save processed data
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(processed_dir / "features.npy", X)
        np.save(processed_dir / "labels.npy", y)
        
        # Save class mapping
        class_mapping = {cls: idx for idx, cls in enumerate(self.target_classes)}
        with open(processed_dir / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f)
        
        print(f"Dataset processed! Shape: {X.shape}")
        return X, y, class_mapping

if __name__ == "__main__":
    preprocessor = ESC50Preprocessor("data/raw")
    X, y, class_mapping = preprocessor.process_dataset()