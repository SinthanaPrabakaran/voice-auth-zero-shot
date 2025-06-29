import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class VoiceDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        self.label_map = {name: idx for idx, name in enumerate(sorted(set(f.split("_")[0] for f in self.file_list)))}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # Load audio
        audio, sr = librosa.load(file_path, sr=None)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Resize to (40, 100)
        if mfcc.shape[1] < 100:
            mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :100]

        # Convert to tensor with shape (1, 40, 100)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        # Get label from filename like 'alice_01.wav' → 'alice'
        label_name = file_name.split("_")[0]
        label = self.label_map[label_name]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mfcc_tensor, label_tensor
