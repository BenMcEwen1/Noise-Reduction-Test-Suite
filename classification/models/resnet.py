# Pipeline implementation for integration with the annotation tool
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score 


class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transform, target_rate=16000, val=False):
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_rate = target_rate
        self.resize = transforms.Resize((224,224))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        resample = torchaudio.transforms.Resample(sr, self.target_rate)
        signal = resample(signal)
        signal = self.transform(signal)
        signal = self.resize(signal)
        signal = signal.repeat(3,1,1)
        return signal,label

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,1])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,3]

    def _get_validation(self, index):
        return self.annotations.iloc[index,2]

    def _filter_annotations(self, data):
        return data.loc[data['Validation'] == self.val]


def encode_label(label, classes):
    # classes = ['possum', 'false-positive', 'unknown']
    y = torch.zeros(len(classes))
    y[classes.index(label)] = 1
    return y

def encode(labels, classes):
    target = []
    for label in labels:
        y = [0,0]
        y[classes.index(label)] = 1
        target.append(y)
    target = torch.Tensor(target)
    return target


class UserModel(torch.nn.Module):
    def __init__(self, out_features, classes):
        super(UserModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(out_features,500)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(500,100)
        self.fc = torch.nn.Linear(100,classes)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def create_model(num_classes):
    # Load pretrained model and "requires_grad" to false to prevent training of pretrained weights
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    # Append linear layer to the model
    # num_ftrs = model.fc.in_features # Num of outputs of pretrained model, new layer 2048 -> 3
    num_out = model.fc.out_features
    userModel = UserModel(num_out, num_classes) 
    return model, userModel


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(training, validation, classes, num_classes, num_epochs=30, batch_size=1, early_stopping=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BASE_MODEL_PATH = f"./models/resnet_model_denoised.pth"
    if os.path.exists(BASE_MODEL_PATH):
        print('Load model')
        model = torch.load(BASE_MODEL_PATH)
        model.eval()
    else:
        model, userModel = create_model(num_classes)
        model = torch.nn.Sequential(model,userModel)
    
    model = model.to(device) # Load into GPU memory if available
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    best_acc = 50
    counter = 0

    data_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        running_loss = 0
        running_corrects = 0
        val_running_loss = 0

        corrects = 0
        GT = []
        pred = []

        total = 0
        # print(f'Epoch {epoch}/{num_epochs}')

        # Training loop
        for input, label in data_loader:
            input = input.to(device)
            optimizer.zero_grad()

            outputs = model(input)
            y = encode(label, classes)
            y = y.to(device)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total += len(torch.argmax(y,dim=1))
            corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            running_corrects = 100*corrects/total

            accuracy = running_corrects.item()
            running_loss += loss.item()

        train_loss.append(running_loss)
        train_accuracy.append(accuracy)

        val_running = 0
        val_total = 0
        val_corrects = 0

        for input, label in val_loader:
            input = input.to(device)
            outputs = model(input)
            y = encode(label, classes)
            y = y.to(device)
            loss = criterion(outputs, y)

            val_total += len(torch.argmax(y,dim=1))
            val_corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            val_running = 100*val_corrects/val_total
            val_running_accuracy = val_running.item()
            val_running_loss += loss.item()

            GT.append(list(y[0].cpu()).index(1))
            pred.append(torch.argmax(outputs,dim=1).cpu().item())

        val_loss.append(val_running_loss)
        val_accuracy.append(val_running_accuracy)

    print('resnet')
    print(f'[{epoch + 1}], Training Accuracy: {accuracy:.2f}, Training loss: {running_loss:.2f}, Val Accuracy: {val_running_accuracy:.2f}, Val loss: {val_running_loss:.2f}')
    print('F1: {}'.format(f1_score(pred, GT, average='macro')))
    print('Precision: {}'.format(precision_score(pred, GT, average='macro')))
    print('Recall: {}'.format(recall_score(pred, GT, average='macro')))
    print('--------------------------------------')
        

        # if ((epoch > 2) and early_stopping):
        #     if val_running_loss < val_loss[-2]:
        #         counter = 0
        #     else:
        #         counter += 1
        #         if counter > 2:
        #             print("Early stopping")
        #             break
        
        # if val_running_accuracy > best_acc:
        #     print('Saving model')
        #     best_acc = val_running_accuracy
        #     torch.save(model, BASE_MODEL_PATH)

    return train_loss, train_accuracy, val_loss, val_accuracy


def resnet_possum(rms, state):
    AUDIO_DIR = f"./possum dataset/{rms}/{state}/"
    ANNOTATIONS = f"./classification/annotations.csv"
    classes = ['possum', 'false-positive']
    NUM_CLASSES = len(classes)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    training = AudioDataset(ANNOTATIONS, AUDIO_DIR, mel_spectrogram, target_rate=16000, val=False)
    validation = AudioDataset(ANNOTATIONS, AUDIO_DIR, mel_spectrogram, target_rate=16000, val=True)

    loss, acc, val_loss, val_acc = train(training, validation, classes, NUM_CLASSES)

    return loss, acc, val_loss, val_acc
