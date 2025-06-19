import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import librosa.display

def analyze_dataset(X, y, class_mapping):
    """Analyze the processed dataset"""
    
    # Class distribution
    plt.figure(figsize=(12, 6))
    class_counts = Counter(y)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    
    # Feature correlation heatmap
    plt.subplot(1, 2, 2)
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('data/processed/data_analysis.png')
    plt.show()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(class_mapping)}")
    print(f"Class distribution: {class_counts}")

def visualize_audio_features(audio_file_path):
    """Visualize audio features for a sample file"""
    audio, sr = librosa.load(audio_file_path, sr=22050)
    
    plt.figure(figsize=(15, 10))
    
    # Waveform
    plt.subplot(3, 2, 1)
    plt.plot(audio)
    plt.title('Waveform')
    
    # Spectrogram
    plt.subplot(3, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spectrogram')
    plt.colorbar()
    
    # MFCC
    plt.subplot(3, 2, 3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title('MFCC')
    plt.colorbar()
    
    # Spectral Centroid
    plt.subplot(3, 2, 4)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    plt.plot(spectral_centroids)
    plt.title('Spectral Centroid')
    
    # Zero Crossing Rate
    plt.subplot(3, 2, 5)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    plt.plot(zcr)
    plt.title('Zero Crossing Rate')
    
    # Frequency Domain
    plt.subplot(3, 2, 6)
    fft = np.abs(np.fft.fft(audio))
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    plt.plot(freqs[:len(freqs)//2], fft[:len(fft)//2])
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('data/processed/audio_features_visualization.png')
    plt.show()