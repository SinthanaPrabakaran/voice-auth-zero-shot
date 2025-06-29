import torch
import librosa
import numpy as np
from cnn_embedding import CNNEmbeddingModel  # assuming you're using embedding model

def extract_embedding(audio_path, model_path='models/cnn_model.pth'):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 40, time)

    model = CNNEmbeddingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        embedding = model(mfcc).numpy()

    return embedding
