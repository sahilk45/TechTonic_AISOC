// Privacy-Enhanced Audio Classifier

// Global variables
let session = null;
let modelLoaded = false;
let audioContext = null;

// Enhanced model parameters
const MODEL_PARAMS = {
    sampleRate: 22050,
    duration: 5,
    nMfcc: 40,
    nFft: 2048,
    hopLength: 512
};

// Your exact category mappings
const CATEGORIES = {
    0: 'Animals',
    1: 'Environment', 
    2: 'Vehicles',
    3: 'Voice'
};

const SUBCATEGORIES = {
    0: 'bike', 1: 'bus', 2: 'car', 3: 'cat', 4: 'crowd',
    5: 'dog', 6: 'elephant', 7: 'horse', 8: 'lion',
    9: 'person_voice', 10: 'rainfall', 11: 'siren',
    12: 'traffic', 13: 'truck'
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateStatus('🔒 Privacy-first audio classifier ready. Click "Load Enhanced Privacy Model" to begin.', 'info');
});

function setupEventListeners() {
    document.getElementById('loadModel').addEventListener('click', loadPrivacyEnhancedModel);
    document.getElementById('classifyBtn').addEventListener('click', classifyAudioPrivacyFirst);
    document.getElementById('testModel').addEventListener('click', testWithSamplePrivacy);
    document.getElementById('audioFile').addEventListener('change', handleFileSelect);
}

// Enhanced privacy-first model loading
async function loadPrivacyEnhancedModel() {
    const loadBtn = document.getElementById('loadModel');
    const classifyBtn = document.getElementById('classifyBtn');
    const testBtn = document.getElementById('testModel');
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<div class="loading"></div> Loading Enhanced Model...';
    
    try {
        updateStatus('🔒 Initializing privacy-first enhanced model...', 'info');
        
        // Setup privacy-preserving cache
        const CACHE_NAME = 'privacy-audio-enhanced-v1';
        try {
            const cache = await caches.open(CACHE_NAME);
            await cache.add('./privacy_first_audio_classifier.onnx');
            updateStatus('📥 Enhanced model cached for offline privacy', 'info');
        } catch (cacheError) {
            console.warn('Caching failed, proceeding with direct load');
        }
        
        // Configure ONNX Runtime for maximum privacy and performance
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime not loaded. Check if the script tag is correct.');
        }
        
        ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
        ort.env.logLevel = 'error';
        ort.env.wasm.simd = true;
        
        updateStatus('🔄 Loading enhanced privacy model (922K parameters)...', 'info');
        
        // Load the enhanced model
        session = await ort.InferenceSession.create('./privacy_first_audio_classifier.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false,
            enableMemPattern: false
        });
        
        const inputInfo = session.inputNames[0];
        const outputInfo = session.outputNames;
        
        // Show model information
        document.getElementById('modelInfo').style.display = 'block';
        
        updateStatus(
            `✅ Enhanced Privacy Model Loaded Successfully!<br>
             🔒 All processing happens client-side only<br>
             📊 Model: Advanced CNN-RNN Hybrid (922K params)<br>
             📥 Input: ${inputInfo}<br>
             📤 Outputs: ${outputInfo.join(', ')}<br>
             🎯 Ready for enhanced private classification!`, 
            'success'
        );
        
        modelLoaded = true;
        classifyBtn.disabled = false;
        testBtn.disabled = false;
        loadBtn.innerHTML = 'Enhanced Model Loaded ✓';
        
    } catch (error) {
        console.error('Enhanced model loading failed:', error);
        updateStatus(`❌ Enhanced model loading failed: ${error.message}`, 'error');
        loadBtn.disabled = false;
        loadBtn.innerHTML = 'Load Enhanced Privacy Model';
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && modelLoaded) {
        updateStatus(`📁 Privacy-first processing ready for: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info');
    }
}

// Enhanced audio classification with privacy protection
async function classifyAudioPrivacyFirst() {
    if (!modelLoaded) {
        updateStatus('❌ Please load the enhanced model first', 'error');
        return;
    }
    
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    
    if (!file) {
        updateStatus('❌ Please select an audio file first', 'error');
        return;
    }
    
    const classifyBtn = document.getElementById('classifyBtn');
    classifyBtn.disabled = true;
    classifyBtn.innerHTML = '<div class="loading"></div> Processing Privately...';
    
    try {
        updateStatus('🔒 Processing audio file privately (client-side only)...', 'info');
        
        // Process audio with enhanced privacy-first extraction
        const mfccFeatures = await processAudioToEnhancedMFCC(file);
        
        updateStatus('🧠 Running enhanced model inference (private)...', 'info');
        
        // Run inference with enhanced model
        const results = await runEnhancedInference(mfccFeatures);
        
        // Display enhanced results
        displayEnhancedResults(results, file.name);
        
    } catch (error) {
        console.error('Privacy-first classification failed:', error);
        updateStatus(`❌ Classification failed: ${error.message}`, 'error');
    } finally {
        classifyBtn.disabled = false;
        classifyBtn.innerHTML = 'Classify Audio';
    }
}

// Enhanced privacy-first audio processing
async function processAudioToEnhancedMFCC(file) {
    console.log(`🔒 Privacy-first processing: ${file.name} (${file.size} bytes)`);
    
    // Initialize audio context with privacy settings
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: MODEL_PARAMS.sampleRate
        });
    }
    
    try {
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        console.log(`🎵 Private audio processing: ${audioBuffer.duration}s, ${audioBuffer.sampleRate}Hz`);
        
        let audioData = audioBuffer.getChannelData(0);
        
        // Client-side resampling if needed
        if (audioBuffer.sampleRate !== MODEL_PARAMS.sampleRate) {
            console.log(`🔄 Private resampling: ${audioBuffer.sampleRate}Hz → ${MODEL_PARAMS.sampleRate}Hz`);
            audioData = await resampleAudioAsync(audioData, audioBuffer.sampleRate, MODEL_PARAMS.sampleRate);
        }
        
        // Ensure exact 5-second duration (client-side only)
        const targetLength = MODEL_PARAMS.sampleRate * MODEL_PARAMS.duration;
        let processedAudio = new Float32Array(targetLength);
        
        if (audioData.length > targetLength) {
            processedAudio.set(audioData.slice(0, targetLength));
        } else {
            processedAudio.set(audioData);
        }
        
        // Extract features using enhanced privacy-first algorithm
        const mfccFeatures = await extractEnhancedPrivacyMFCC(processedAudio);
        
        console.log('✅ Enhanced privacy-first processing completed');
        return mfccFeatures;
        
    } catch (error) {
        console.error('Privacy-first audio processing failed:', error);
        throw new Error(`Enhanced private processing failed: ${error.message}`);
    }
}

// Enhanced MFCC extraction for privacy and accuracy
async function extractEnhancedPrivacyMFCC(audioData) {
    const nFft = MODEL_PARAMS.nFft;
    const hopLength = MODEL_PARAMS.hopLength;
    const nMfcc = MODEL_PARAMS.nMfcc;
    const expectedFrames = 216;
    
    console.log('🔄 Enhanced privacy MFCC extraction starting...');
    
    // Pre-emphasis filter for better frequency analysis
    const preEmphasized = new Float32Array(audioData.length);
    preEmphasized[0] = audioData[0];
    for (let i = 1; i < audioData.length; i++) {
        preEmphasized[i] = audioData[i] - 0.97 * audioData[i - 1];
    }
    
    // Hamming window for optimal spectral analysis
    const window = new Float32Array(nFft);
    for (let i = 0; i < nFft; i++) {
        window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (nFft - 1));
    }
    
    const mfccMatrix = new Float32Array(nMfcc * expectedFrames);
    
    // Enhanced spectral analysis with progressive processing
    for (let frame = 0; frame < expectedFrames; frame++) {
        const startIdx = frame * hopLength;
        
        // Extract windowed frame
        const frameData = new Float32Array(nFft);
        for (let i = 0; i < nFft; i++) {
            if (startIdx + i < preEmphasized.length) {
                frameData[i] = preEmphasized[startIdx + i] * window[i];
            }
        }
        
        // Enhanced power spectrum computation
        const powerSpectrum = new Float32Array(nFft / 2);
        for (let k = 0; k < nFft / 2; k++) {
            let real = 0, imag = 0;
            const step = Math.max(1, Math.floor(nFft / 512));
            
            for (let n = 0; n < nFft; n += step) {
                const angle = -2 * Math.PI * k * n / nFft;
                real += frameData[n] * Math.cos(angle);
                imag += frameData[n] * Math.sin(angle);
            }
            
            powerSpectrum[k] = Math.log(real * real + imag * imag + 1e-10);
        }
        
        // Enhanced mel filterbank and DCT
        for (let mfccBin = 0; mfccBin < nMfcc; mfccBin++) {
            let energy = 0;
            
            // Improved mel-scale frequency mapping
            const melFreq = 2595 * Math.log10(1 + mfccBin * 8000 / (2 * nMfcc * 700));
            const binStart = Math.floor(melFreq * powerSpectrum.length / 22050);
            const binEnd = Math.min(binStart + Math.floor(powerSpectrum.length / nMfcc), powerSpectrum.length);
            
            for (let bin = binStart; bin < binEnd; bin++) {
                const weight = 1 - Math.abs(bin - (binStart + binEnd) / 2) / ((binEnd - binStart) / 2 + 1e-8);
                energy += powerSpectrum[bin] * Math.max(0, weight);
            }
            
            // Enhanced DCT for final MFCC coefficient
            let mfccValue = 0;
            for (let k = 0; k < Math.min(nMfcc, 13); k++) {
                mfccValue += energy * Math.cos(Math.PI * mfccBin * (k + 0.5) / nMfcc);
            }
            
            mfccMatrix[frame * nMfcc + mfccBin] = mfccValue;
        }
        
        // Progress updates with privacy assurance
        if (frame % 40 === 0) {
            const progress = Math.round((frame / expectedFrames) * 100);
            updateStatus(`🔒 Private processing... ${progress}% (client-side only)`, 'info');
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    
    // Enhanced normalization for better model performance
    const mean = mfccMatrix.reduce((a, b) => a + b, 0) / mfccMatrix.length;
    let variance = mfccMatrix.reduce((a, b) => a + (b - mean) ** 2, 0) / mfccMatrix.length;
    variance = Math.max(variance, 0.01);
    const std = Math.sqrt(variance);
    
    for (let i = 0; i < mfccMatrix.length; i++) {
        mfccMatrix[i] = (mfccMatrix[i] - mean) / std;
    }
    
    console.log('✅ Enhanced privacy MFCC extraction completed');
    return mfccMatrix;
}

// Enhanced model inference
async function runEnhancedInference(mfccFeatures) {
    try {
        const nFrames = mfccFeatures.length / MODEL_PARAMS.nMfcc;
        
        // Reshape for enhanced model input
        const reshapedFeatures = new Float32Array(1 * MODEL_PARAMS.nMfcc * nFrames * 1);
        
        // Copy data in correct order for the enhanced model
        for (let frame = 0; frame < nFrames; frame++) {
            for (let mfcc = 0; mfcc < MODEL_PARAMS.nMfcc; mfcc++) {
                reshapedFeatures[mfcc * nFrames + frame] = mfccFeatures[frame * MODEL_PARAMS.nMfcc + mfcc];
            }
        }
        
        // Create input tensor with exact shape for enhanced model
        const inputTensor = new ort.Tensor('float32', reshapedFeatures, [1, MODEL_PARAMS.nMfcc, nFrames, 1]);
        
        console.log(`🔒 Running enhanced private inference with shape: ${inputTensor.dims}`);
        
        // Run inference with enhanced model
        const results = await session.run({ [session.inputNames[0]]: inputTensor });
        
        // Process enhanced model outputs
        const categoryOutput = results[session.outputNames[0]].data;
        const subcategoryOutput = results[session.outputNames[1]].data;
        
        const categoryPred = Array.from(categoryOutput).indexOf(Math.max(...categoryOutput));
        const subcategoryPred = Array.from(subcategoryOutput).indexOf(Math.max(...subcategoryOutput));
        
        return {
            category: categoryPred,
            subcategory: subcategoryPred,
            categoryConfidence: Math.max(...categoryOutput),
            subcategoryConfidence: Math.max(...subcategoryOutput),
            categoryProbs: Array.from(categoryOutput),
            subcategoryProbs: Array.from(subcategoryOutput),
            privacy: 'guaranteed_client_side',
            model: 'enhanced_922k_params'
        };
        
    } catch (error) {
        console.error('Enhanced inference failed:', error);
        throw error;
    }
}

// Enhanced results display with privacy information
function displayEnhancedResults(results, filename) {
    const resultsDiv = document.getElementById('results');
    
    const categoryName = CATEGORIES[results.category] || `Category ${results.category}`;
    const subcategoryName = SUBCATEGORIES[results.subcategory] || `Subcategory ${results.subcategory}`;
    
    resultsDiv.innerHTML = `
        <div class="results">
            <h3>🎯 Enhanced Privacy-First Results for: ${filename}</h3>
            
            <div style="background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <small>🔒 <strong>Privacy Confirmed:</strong> All processing completed client-side. No data transmitted.</small><br>
                <small>🧠 <strong>Enhanced Model:</strong> 922K parameter CNN-RNN hybrid architecture</small>
            </div>
            
            <div style="margin: 20px 0;">
                <h4>📊 Category: ${categoryName}</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${results.categoryConfidence * 100}%"></div>
                    <div class="confidence-text">${(results.categoryConfidence * 100).toFixed(2)}%</div>
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                <h4>🏷️ Subcategory: ${subcategoryName}</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${results.subcategoryConfidence * 100}%"></div>
                    <div class="confidence-text">${(results.subcategoryConfidence * 100).toFixed(2)}%</div>
                </div>
            </div>
            
            <details style="margin-top: 20px;">
                <summary><strong>📈 Detailed Probabilities (Enhanced Model)</strong></summary>
                <div style="margin-top: 10px;">
                    <h5>Category Probabilities:</h5>
                    ${results.categoryProbs.map((prob, idx) => 
                        `<div style="margin: 5px 0;">
                            ${CATEGORIES[idx] || `Category ${idx}`}: 
                            <strong>${(prob * 100).toFixed(2)}%</strong>
                            <div style="width: ${prob * 200}px; height: 8px; background: linear-gradient(90deg, #4CAF50, #2196F3); border-radius: 4px; display: inline-block; margin-left: 10px;"></div>
                        </div>`
                    ).join('')}
                    
                    <h5 style="margin-top: 15px;">Top Subcategory Probabilities:</h5>
                    ${results.subcategoryProbs
                        .map((prob, idx) => ({ prob, idx }))
                        .sort((a, b) => b.prob - a.prob)
                        .slice(0, 5)
                        .map(item => 
                            `<div style="margin: 5px 0;">
                                ${SUBCATEGORIES[item.idx] || `Subcategory ${item.idx}`}: 
                                <strong>${(item.prob * 100).toFixed(2)}%</strong>
                                <div style="width: ${item.prob * 200}px; height: 8px; background: linear-gradient(90deg, #4CAF50, #2196F3); border-radius: 4px; display: inline-block; margin-left: 10px;"></div>
                            </div>`
                        ).join('')}
                </div>
            </details>
        </div>
    `;
    
    updateStatus('✅ Enhanced privacy-first classification completed successfully!', 'success');
}

// Enhanced sample testing with privacy assurance
async function testWithSamplePrivacy() {
    if (!modelLoaded) {
        updateStatus('❌ Please load the enhanced model first', 'error');
        return;
    }
    
    const testBtn = document.getElementById('testModel');
    testBtn.disabled = true;
    testBtn.innerHTML = '<div class="loading"></div> Testing Enhanced Model...';
    
    try {
        updateStatus('🧪 Generating enhanced sample data (private)...', 'info');
        
        // Create sample data for enhanced model testing
        const nFrames = 216;
        const nMfcc = MODEL_PARAMS.nMfcc;
        const sampleData = new Float32Array(1 * nMfcc * nFrames * 1);
        
        // Generate varied sample data for testing
        for (let i = 0; i < sampleData.length; i++) {
            sampleData[i] = (Math.random() - 0.5) * 0.8;
        }
        
        updateStatus('🧠 Running enhanced model test inference (private)...', 'info');
        
        const inputTensor = new ort.Tensor('float32', sampleData, [1, nMfcc, nFrames, 1]);
        const results = await session.run({ [session.inputNames[0]]: inputTensor });
        
        const categoryOutput = results[session.outputNames[0]].data;
        const subcategoryOutput = results[session.outputNames[1]].data;
        
        const categoryPred = Array.from(categoryOutput).indexOf(Math.max(...categoryOutput));
        const subcategoryPred = Array.from(subcategoryOutput).indexOf(Math.max(...subcategoryOutput));
        
        displayEnhancedResults({
            category: categoryPred,
            subcategory: subcategoryPred,
            categoryConfidence: Math.max(...categoryOutput),
            subcategoryConfidence: Math.max(...subcategoryOutput),
            categoryProbs: Array.from(categoryOutput),
            subcategoryProbs: Array.from(subcategoryOutput)
        }, 'Enhanced Model Test Data');
        
    } catch (error) {
        console.error('Enhanced test failed:', error);
        updateStatus(`❌ Enhanced test failed: ${error.message}`, 'error');
    } finally {
        testBtn.disabled = false;
        testBtn.innerHTML = 'Test with Sample';
    }
}

// Audio resampling utility
async function resampleAudioAsync(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) {
        return audioData;
    }
    
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.floor(audioData.length / ratio);
    const resampledData = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        const originalIndex = i * ratio;
        const index = Math.floor(originalIndex);
        const fraction = originalIndex - index;
        
        if (index + 1 < audioData.length) {
            resampledData[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
        } else {
            resampledData[i] = audioData[index];
        }
    }
    
    return resampledData;
}

// Status update utility
function updateStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    const className = type === 'error' ? 'results error' : 
                     type === 'success' ? 'results' : 
                     'results';
    
    statusDiv.innerHTML = `<div class="${className}">${message}</div>`;
}
