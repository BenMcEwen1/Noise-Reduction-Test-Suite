from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import torch 
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score 

torch.cuda.empty_cache()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class UserModel(nn.Module):
    # Create user model for transfer learning
    def __init__(self):
        super(UserModel, self).__init__()
        in_features = 12
        self.linear1 = torch.nn.Linear(in_features, 2)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.softmax(x)
        return x
    
class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, target_rate=16000, val=False):
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        self.audio_dir = audio_dir
        self.target_rate = target_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index) + '.wav'
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        resample = torchaudio.transforms.Resample(sr, self.target_rate)
        signal = resample(signal[0])
        signal = self.feature_extractor(signal, sampling_rate=self.target_rate, return_tensors="pt")
        return signal,label

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, str(self.annotations.iloc[index,0]))
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,2]

    def _get_validation(self, index):
        return self.annotations.iloc[index,3]

    def _filter_annotations(self, data):
        return data.loc[data['Validation'] == self.val]
    

def encode(labels, classes):
    target = []
    for label in labels:
        y = [0,0]
        y[classes.index(label)] = 1
        target.append(y)
    target = torch.Tensor(target)
    return target
    

def train(training_set, validation_set, classes, num_epoch=10, batch_size=5):
    usermodel = UserModel()
    usermodel = usermodel.to(device)

    model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
    model = model.to(device)

    learning_rate = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.parameters()) + list(usermodel.parameters()), lr=learning_rate) # Recommeds: sweep lr [1e-5, 1e-4], Adam optim
    # data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    best_acc = 0

    for epoch in range(num_epoch):

        if (epoch > 5):
            learning_rate = learning_rate/2
            optimizer = optim.SGD(list(model.parameters()) + list(usermodel.parameters()), lr=learning_rate)

        running_loss = 0
        running_corrects = 0
        val_running_loss = 0
        corrects = 0
        total = 0
        i = 0

        for data, label in training_set:
            data = data.to(device)
            optimizer.zero_grad()

            embeddings = model(**data).logits
            outputs = usermodel(embeddings)
            y = encode([label], classes) # list wrapper as label not batched
            y = y.to(device)

            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()

            total += len(torch.argmax(y,dim=1))
            corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            running_corrects = 100*corrects/total

            accuracy = running_corrects.item()
            running_loss += loss.item()

            # if (i % 100 == 1) and (i != 1):
            #     print(f"[{i}/{len(training_set)}] - Training Accuracy: {accuracy:.2f}, Training loss: {running_loss/i:.2f}")
            # i += 1

        train_loss.append(running_loss)
        train_accuracy.append(accuracy)

        val_running = 0
        val_total = 0
        val_corrects = 0

        GT = []
        pred = []

        for data, label in validation_set:
            data = data.to(device)
            embeddings = model(**data).logits
            outputs = usermodel(embeddings)
            
            y = encode([label], classes)
            y = y.to(device)
            loss = criterion(outputs, y)

            val_total += len(torch.argmax(y,dim=1))
            val_corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            val_running = 100*val_corrects/val_total
            val_running_accuracy = val_running.item()
            val_running_loss += loss.item()

            GT.append(list(y[0].cpu()).index(1))
            pred.append(torch.argmax(outputs,dim=1).cpu().item())

            # if (i % 100 == 1) and (i != 1):
            #     print(f"[{i}/{len(training_set)}] - Validation Accuracy: {val_running_accuracy:.2f}, validation loss: {val_running_loss/i:.2f}")
            # i += 1

        val_loss.append(val_running_loss)
        val_accuracy.append(val_running_accuracy)
        
        print('HuBERT')
        print(f'[{epoch + 1}], Training Accuracy: {accuracy:.2f}, Training loss: {running_loss/len(training_set):.2f}, Val Accuracy: {val_running_accuracy:.2f}, Val loss: {val_running_loss/len(validation_set):.2f}')
        print('F1: {}'.format(f1_score(pred, GT, average='macro')))
        print('Precision: {}'.format(precision_score(pred, GT, average='macro')))
        print('Recall: {}'.format(recall_score(pred, GT, average='macro')))
        print('--------------------------------------')

        MODEL_PATH = './classification/models/bird/hubert_model_denoised.pth'
        if (val_running_accuracy > best_acc):
            best_acc = val_running_accuracy
            torch.save(model, MODEL_PATH)
        
    return train_loss, train_accuracy, val_loss, val_accuracy

def hubert():
    AUDIO_DIR = f"./bird dataset/denoised/"
    ANNOTATIONS = f"./freefield1010.csv"
    classes = [1,0]

    # print('Generating dataset...')
    training = AudioDataset(ANNOTATIONS, AUDIO_DIR, target_rate=16000, val=False)
    validation = AudioDataset(ANNOTATIONS, AUDIO_DIR, target_rate=16000, val=True)

    # print('training...')
    loss, acc, val_loss, val_acc = train(training, validation, classes)
