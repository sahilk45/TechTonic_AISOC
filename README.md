# 🔊 AISOC 2025 - Acoustic Event Detection System

## 📌 Project Title:  
**Acoustic Source and Event Detection: Privacy-First Mobile/Web Application**

## 👥 Team Information

**Team Name:** _TechTonic_  
**Project Number:** 2

### 👨‍💻 Team Members

| Name              | Email                       |
|-------------------|-----------------------------|
|   SAHIL KUMAR     | sahilkumarb527@gmail.com    |
| ABHISHEK          | abhishekgoyal9728@gmail.com |
| SAKSHAM TOMAR     | tomarsaksham2006@gmail.com  |
| SUBOM SUKLABAIDYA | subomsuklabaidya@gmail.com  |

## 📖 Problem Statement

> **A privacy-first mobile/web application that detects and identifies environmental sounds such as alarms and sirens in real time.**  

This project is based on **DCASE 2025 Task 4** and aims to enable real-time acoustic event detection while ensuring user privacy. The application runs with a cloud-based backend for ML processing and provides intelligent alerts for emergency and safety-critical sounds.

## 🏗️ System Architecture

Our solution consists of two main components:

### 🖥️ **Backend (FastAPI + TensorFlow)**
- **Deployment**: [https://event-detection.onrender.com](https://event-detection.onrender.com)
- **Technology Stack**: Python 3.11, FastAPI, TensorFlow, librosa
- **Features**:
  - Audio processing and MFCC feature extraction
  - TensorFlow native model (.keras/.pkl format) for predictions
  - Smart alert system with 70% confidence threshold
  - Support for multiple audio formats (WAV, MP3, etc.)
  - CORS-enabled for cross-origin requests

### 🌐 **Frontend (React)**
- **Deployment**: [https://event-detection-frontened.onrender.com](https://event-detection-frontened.onrender.com)
- **Technology Stack**: React.js, Axios, modern web APIs
- **Features**:
  - Drag-and-drop audio file upload
  - Real-time audio recording capabilities
  - Interactive results display
  - Alert notifications for dangerous events

## 🎯 Key Features

### 🚨 **Smart Alert System**
- **Emergency Sounds**: Fire alarms, sirens, smoke alarms
- **Safety Events**: Glass breaking, baby crying, screaming
- **Security Events**: Dog barking, door breaking, footsteps
- **Vehicle Emergencies**: Car alarms, crashes, horns

### 🎵 **Audio Classification Categories**
- **Animals**: Dog, Cat, Bird, Other Animals
- **Environment**: Rain, Wind, Thunder, Water
- **Vehicles**: Car, Truck, Motorcycle, Aircraft  
- **Voice**: Male Voice, Female Voice, Child Voice, Crowd

### ⚡ **Technical Capabilities**
- Real-time audio processing with 5-second windows
- MFCC feature extraction (40 coefficients)
- Multi-output model supporting both category and subcategory predictions
- Memory-optimized deployment with 512MB RAM support( Yet to be Optimized more for complete functioning)

## 🚀 Quick Start Guide

### 🔧 Local Development Setup

#### Prerequisites
- Node.js (v14+) & npm
- Python 3.11+
- Git and Git LFS (for large model files)

#### 1. Clone the Repository
```bash
git clone https://github.com/sahilk45/TechTonic_AISOC.git
cd TechTonic_AISOC
```


#### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

#### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

```

#### 4. Access the Application on your local machine
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI 0.116.1
- **ML Framework**: TensorFlow 
- **Audio Processing**: librosa 0.11.0
- **Data Processing**: NumPy, SciPy, pandas
- **Deployment**: Docker on Render

### Frontend  
- **Framework**: React.js
- **HTTP Client**: Axios
- **Styling**: Modern CSS/Bootstrap
- **Deployment**: Static site on Render

### Model Details
- **Architecture**: Convolutional Neural Network
- **Input**: MFCC features (40 x 216 x 1)
- **Output**: Multi-class classification with confidence scores
- **Format**: TensorFlow native (.keras/.pkl) for version compatibility
- **Size**: ~130MB (managed with Git LFS)

## 📊 Performance Metrics

- **Confidence Threshold**: 70% for alert triggering
- **Processing Time**: ~2-3 seconds per 5-second audio clip
- **Memory Usage**: <512MB (optimized for free-tier deployment)
- **Supported Formats**: WAV, MP3, M4A, FLAC


## 🔗 Live Deployment

### Production URLs
- **🌐 Frontend Application**: [https://event-detection-frontened.onrender.com](https://event-detection-frontened.onrender.com)
- **🔧 Backend API**: [https://event-detection.onrender.com](https://event-detection.onrender.com)
- **📚 API Documentation**: [https://event-detection.onrender.com/docs](https://event-detection.onrender.com/docs)

### Deployment Features
- **Auto-deployment** from GitHub main branch
- **Git LFS** support for large model files
- **CORS** properly configured for cross-origin requests
- **Health checks** and monitoring enabled
- **Environment variables** for production configuration

## 🔄 Development Workflow

### Model Development Process
1. **Data Preprocessing**: Audio → MFCC features using librosa
2. **Model Training**: TensorFlow/Keras with tf-nightly locally
3. **Model Conversion**: Pickle → TensorFlow native format for deployment
4. **Version Compatibility**: Ensuring model works across TF versions
5. **Git LFS**: Managing large model files (100MB+)

### Deployment Pipeline
1. **Code Push**: GitHub repository with automatic deployments
2. **Backend Build**: Docker container with Python dependencies
3. **Frontend Build**: Static site generation and deployment
4. **Integration Testing**: CORS and API endpoint verification

## 🐛 Troubleshooting

### Common Issues & Solutions

**CORS Errors:**
- Ensure frontend URL is added to backend CORS origins
- Check browser console for specific error messages

**Model Loading Issues:**
- Verify .keras/.pkl file exists and is properly committed with Git LFS
- Check TensorFlow version compatibility

**Memory Issues:**
- Monitor Render service memory usage
- Consider upgrading to paid plan for tf-nightly usage

**Audio Processing Errors:**
- Verify audio file format is supported
- Check file size limits (usually <10MB)

## 📈 Future Enhancements

- [ ] Real-time audio streaming support properly working
- [ ] Mobile app development (React Native)
- [ ] Offline model deployment (TensorFlow.js)
- [ ] Custom alert configuration
- [ ] Integration with IoT devices if possible somewhere with mentors
- [ ] Enhanced privacy features

## 📄 License

This project is developed for **AISOC 2025** educational purposes and follows academic use guidelines.

## 🙏 Acknowledgments

- **DCASE 2025** for providing the foundational task framework
- **AISOC 2025** organizing committee of PClub
- **Render** for cloud deployment services
- **Open-source community** for TensorFlow, React, and FastAPI

### ✨ Happy Coding from Team TechTonic! 🚀

**Built with ❤️ by Team TechTonic for AISOC 2025**

