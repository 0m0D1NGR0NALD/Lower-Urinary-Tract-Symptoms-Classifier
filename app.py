import gradio as gr
import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

dataset = pd.read_csv('UTIv2.csv')
dataset = dataset.drop('filename',axis=1)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
sc = sc.fit(x_train)

model = joblib.load('UTI.pkl')

def predictor(audio_filename):
    y, sr = librosa.load(audio_filename, mono=True, duration=5)

    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    v = []
    for e in mfcc:
        v.append(np.mean(e))
    mfcc1 = v[0]
    mfcc2 = v[1]
    mfcc3 = v[2]
    mfcc4 = v[3]
    mfcc5 = v[4]
    mfcc6 = v[5]
    mfcc7 = v[6]
    mfcc8 = v[7]
    mfcc9 = v[8]
    mfcc10 = v[9]
    mfcc11 = v[10]
    mfcc12 = v[11]
    mfcc13 = v[12]
    mfcc14 = v[13]
    mfcc15 = v[14]
    mfcc16 = v[15]
    mfcc17 = v[16]
    mfcc18 = v[17]
    mfcc19 = v[18]
    mfcc20 = v[19]

    features = np.array([[chroma_stft,rmse,spec_cent,spec_bw,rolloff,zcr,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,mfcc13,mfcc14,mfcc15,mfcc16,mfcc17,mfcc18,mfcc19,mfcc20]])
    
    prediction = model.predict(sc.transform(features))
    
    if prediction[0] == 1:
        result = 'Normal'
    else:
        result = 'Infected'
    return result

app = gr.Interface(fn=predictor,
inputs=gr.Audio(source="upload",type="filepath",label="Please Upload Audio file here:"),
outputs=gr.Textbox(label="Result"),title="SMART LUTS DETECTOR",description="UTI Prediction Model",examples=[["normal 1_rn.wav"]])
app.launch()