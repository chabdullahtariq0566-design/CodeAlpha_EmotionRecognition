# 🎙️ Task 2 — Emotion Recognition from Speech
### CodeAlpha Machine Learning Internship

---

## 📌 Objective
Recognize human emotions — **angry, happy, sad, neutral** — directly from raw speech audio files using signal processing and deep learning techniques.

---

## 🧠 Problem Statement
Human speech carries rich emotional signals beyond just words. Automatic emotion recognition from audio has applications in mental health monitoring, customer service, virtual assistants, and human-computer interaction. This project builds a pipeline that extracts acoustic features from audio and classifies the speaker's emotional state.

---

## 📂 Dataset

| Source | Details |
|--------|---------|
| **Primary** | RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song |
| **Kaggle Link** | `uwrfkaggler/ravdess-emotional-speech-audio` |
| **Fallback** | Synthetic feature vectors (auto-generated if RAVDESS not found) |
| **Alternatives** | TESS, EMO-DB (can extend the loader) |

### RAVDESS Emotion Codes
| Code | Emotion | Mapped To |
|------|---------|-----------|
| 01 | Neutral | neutral |
| 02 | Calm | neutral |
| 03 | Happy | happy |
| 04 | Sad | sad |
| 05 | Angry | angry |
| 06 | Fearful | sad |
| 07 | Disgust | angry |
| 08 | Surprised | happy |

> Grouped into 4 core classes: **angry, happy, sad, neutral**

---

## ⚙️ Approach & Methodology

```
Audio Files (.wav)
   ↓
Librosa Audio Loading (sr=22050 Hz, duration=3s)
   ↓
Feature Extraction per file:
   ├── MFCC           (40 coefficients × mean + std = 80)
   ├── Chroma STFT    (12 × mean + std = 24)
   ├── Mel Spectrogram(128 × mean + std = 256)
   ├── Zero Crossing Rate (mean + std = 2)
   ├── RMS Energy     (mean + std = 2)
   ├── Spectral Centroid (mean + std = 2)
   └── Spectral Rolloff  (mean + std = 2)
   Total: ~440 features per audio file
   ↓
Label Encoding (angry=0, happy=1, neutral=2, sad=3)
   ↓
StandardScaler normalization
   ↓
Train / Test Split (80/20, stratified)
   ↓
Classical ML Models (SVM, RF, GBM)
   +
CNN Model (Conv1D on feature sequence)
   ↓
Evaluation & Visualisation
```

---

## 🔊 Feature Extraction Details

### MFCCs (Mel-Frequency Cepstral Coefficients)
The most important features for speech emotion recognition. They represent the short-term power spectrum of audio on a mel scale — capturing the timbral texture of voice.

### Chroma Features
Represent the energy distribution across 12 pitch classes. Useful for capturing tonal qualities that differ across emotions.

### Mel Spectrogram
A spectrogram where frequencies are converted to the mel scale — mimicking human auditory perception.

### Zero Crossing Rate (ZCR)
How often the signal changes sign — higher in noisy/angry speech, lower in calm/sad speech.

### RMS Energy
Root Mean Square energy — higher for loud emotional states (angry), lower for quiet ones (sad).

---

## 🤖 Models Used

### Classical ML
| Model | Key Hyperparameters |
|-------|-------------------|
| **SVM (RBF kernel)** | C=10, gamma='scale' |
| **Random Forest** | n_estimators=200 |
| **Gradient Boosting** | n_estimators=150, lr=0.1 |

### Deep Learning — CNN (Conv1D)
```
Input (440, 1)
   → Conv1D(64) + BatchNorm + MaxPool + Dropout(0.25)
   → Conv1D(128) + BatchNorm + MaxPool + Dropout(0.25)
   → Conv1D(256) + GlobalAveragePooling + Dropout(0.3)
   → Dense(128) + Dropout(0.4)
   → Dense(4, softmax)
```
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping + ReduceLROnPlateau

---

## 📊 Evaluation Metrics
- **Accuracy** — Overall emotion classification accuracy
- **F1-Score (Macro)** — Averaged F1 across all 4 emotion classes
- **Classification Report** — Per-emotion Precision, Recall, F1
- **Confusion Matrix** — Which emotions are confused with which

---

## 📈 Expected Results

| Model | Accuracy | F1-Macro |
|-------|----------|---------|
| SVM (RBF) | ~65–72% | ~0.64–0.71 |
| Random Forest | ~62–68% | ~0.61–0.67 |
| Gradient Boosting | ~64–70% | ~0.63–0.69 |
| **CNN (Conv1D)** | **~70–80%** | **~0.69–0.79** |

> Note: Results vary based on real vs. synthetic data. RAVDESS with real audio gives best results.

---

## 📦 Output Files Generated
- `task2_features.png` — Mean MFCC per emotion + class distribution
- `task2_results.png` — CNN training curves, model comparison, confusion matrix

---

## 🚀 How to Run on Kaggle

1. Go to [kaggle.com](https://kaggle.com) → **+ New Notebook**
2. Add dataset: **+ Add Data** → search `ravdess-emotional-speech-audio`
3. Enable GPU: **Settings → Accelerator → GPU T4**
4. Paste `task2_emotion_recognition.py` into a cell
5. Click **Run All** (~5–10 minutes with GPU)

> ⚠️ Without RAVDESS added, the notebook runs on synthetic data (still demonstrates the full pipeline).

---

## 🛠️ Libraries Required
```
numpy, pandas, matplotlib, seaborn
librosa          (pre-installed on Kaggle)
scikit-learn
tensorflow >= 2.x
```

---

## 📁 GitHub Repository
Name your repo: `CodeAlpha_EmotionRecognition`

---

*CodeAlpha Machine Learning Internship — Task 2 of 4*
