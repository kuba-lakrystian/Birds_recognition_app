import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import librosa
import numpy as np
import pandas as pd
import keras
import base64
import shutil
from io import BytesIO

app = dash.Dash(__name__)
server = app.server

def load_files():
    global selector
    global ss
    global lb
    global model
    
    selector = pd.read_pickle(r'selector.pickle')
    ss = pd.read_pickle(r'ss.pickle')
    lb = pd.read_pickle(r'lb.pickle')
    model = keras.models.load_model('model_ann.h5')

def extract_features(content_string, filename):
  try:
      decoded = base64.b64decode(content_string)
      with open('dummy' + filename, 'wb') as file: # this is really annoying!
          shutil.copyfileobj(BytesIO(decoded), file, length=131072)
      X, sample_rate = librosa.load('dummy' + filename, mono=True)
      os.remove('dummy' + filename)
    
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
      stft = np.abs(librosa.stft(X))
      chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
      mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
      contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
      tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
      sr=sample_rate).T,axis=0)
  
  except ZeroDivisionError:
      mfccs = np.zeros(40)
      chroma = np.zeros(12)
      mel = np.zeros(128)
      contrast = np.zeros(7)
      tonnetz = np.zeros(6)

  return mfccs, chroma, mel, contrast, tonnetz

def prepare_prediction(data):
    data_pruned = selector.transform(data)
    data_pruned = ss.transform(data_pruned)
            
    predictions = model.predict_classes(data_pruned)
    predictions = lb.inverse_transform(predictions)
    
    return predictions
    
app.layout = html.Div(
    [
        html.H1("Wybór plików"),
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Wybierz plik dźwiękowy z dźwiękiem ptaka"]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.H2("Twój ptak to:"),
        html.Ul(id="file-list"),
        html.Ul(id="drugi"),
    ],
    style={"max-width": "500px"},
)


@app.callback(
    [Output("file-list", "children"), Output('drugi', "children")],
    [Input("upload-data", "filename"), Input("upload-data", "contents")]
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        
        load_files()
        
        content_type, content_string = uploaded_file_contents.split(',')
        train_features = extract_features(content_string, uploaded_filenames)
        
        features_train = []
            
        features_train.append(np.concatenate((
                train_features[0],
                train_features[1], 
                train_features[2], 
                train_features[3],
                train_features[4]), axis=0))
            
        X_train = np.array(features_train)
            
        predictions_mod = prepare_prediction(X_train)
            
        return [html.Li(predictions_mod), html.Li(uploaded_filenames)]            
    else:
        return [html.Li("Nie wybrano pliku dźwiękowego"), html.Li("Załaduj dźwięk")]

if __name__ == "__main__":
    app.run_server()
    
