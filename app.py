import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import torch 
import numpy as np

from torch import nn, optim

import torch.nn.functional as F

st.markdown('<style>body{background-color: #92a8d1;}</style>',unsafe_allow_html=True)

st.title("ECG anomaly detection to predict abnormal heart beat")

page_bg_img = '''
<style>
body {
background-image: url("https://p4.wallpaperbetter.com/wallpaper/125/548/31/ecg-ekg-minimalism-music-wallpaper-preview.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len,n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features


class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x


def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum')
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

uploaded_file = st.file_uploader("Upload Files",type=['csv','txt'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    df=pd.read_csv(uploaded_file)
    st.write(df)
    fig = px.line(df.iloc[0,:])
    st.write(fig)

#df=pd.read_csv(uploaded_file,header=None, delim_whitespace=True)
import dill 
butt = st.button('PREDICT')
THRESHOLD = 26
if butt:
    model = torch.load(r'D:\Education\Project\Final\model.pth',map_location=torch.device('cpu'))
    
        
    
    tensor_df,a,b = create_dataset(df)
    _, loss = predict(model, tensor_df)
    if loss[0]>THRESHOLD:
        
        st.markdown(
        f'<div style="color: red; font-family:verdana;font-weight:bold; font-size: 40px">Heartbeat is ABNORMAL</div>',
        unsafe_allow_html=True)
    else:
        
        st.markdown(
        f'<div style="color: green;font-family:verdana;font-weight:bold; font-size: 40px">Heartbeat is NORMAL</div>',
        unsafe_allow_html=True)
 
    
    
    
    