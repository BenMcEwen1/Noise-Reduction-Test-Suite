import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torchaudio
import os
import pandas as pd
    
class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, target_rate=16000, val=False):
        super(AudioDataset, self).__init__()
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        self.audio_dir = audio_dir
        self.target_rate = target_rate
        self.resize = transforms.Resize((224,224))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        resample = torchaudio.transforms.Resample(sr, self.target_rate)
        signal = resample(signal[0])
        feature = self.feature_extractor(signal, sampling_rate=self.target_rate, return_tensors="pt")

        with torch.no_grad():
            feature = feature['input_values']
            embeddings = self.classifier(feature)

        return embeddings,feature

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,1])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,3]

    def _get_validation(self, index):
        return self.annotations.iloc[index,2]

    def _filter_annotations(self, data):
        return data.loc[data['Validation'] == self.val]
    
    def shuffle(self):
        pass
