# -*- coding: utf-8 -*-

## Importamos los módulos necesarios
import numpy as np
import pandas as pd
import torch.optim as optim
from modelo import SuperResolutionNet 
import argparse
import os                    # nn.BatchNorm2d(2,affine=False),
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py  
import matplotlib.pyplot as plt

## Chequeo del uso de varias GPUs
n_gpus = torch.cuda.device_count()
print(f"Número de GPUs: {n_gpus}")
device = torch.device("cuda")

## Parser 
parser = argparse.ArgumentParser() 
parser.add_argument('--epoch_num', type=int, default=25)
args = parser.parse_args() 

epoch_num = args.epoch_num
epoch_num_char = str(epoch_num)

num_workers = 0

## Definición de rutas y creación de directorios
datapath  = "/lhome/ext/i3m121/i3m1213/data/h5yentrenamiento/"
modelpath_test = "/lhome/ext/i3m121/i3m1213/barrido/barrido_hiperparámetros/definitivo1/"+"epoch-"+epoch_num_char+".pth"
savepath  = "/lhome/ext/i3m121/i3m1213/barrido/barrido_hiperparámetros/definitivo1/test_fastMRI/"+"epoca"+epoch_num_char
os.makedirs(savepath, exist_ok=True)

## Clase para la obtención de imágenes a alta y baja resolución 
class prepareData_test(Dataset):
    def __init__(self, train_or_test):
        self.files = os.listdir(datapath+train_or_test)
        self.train_or_test= train_or_test
    def __len__(self):
        return len(self.files) 
    def __getitem__(self, idx):
        filename = datapath+self.train_or_test+'/'+self.files[idx]
        with h5py.File(filename, 'r') as f:
            data = f['data'][:]
            target = f['target'][:]
        return data, target 
   

## Conjunto test
testset = prepareData_test('test')
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=num_workers,pin_memory=True)

## Dim que hemos impuesto en ed_wu_prepare_train_data2
z_hr,x_hr,y_hr = 56, 240, 240 # son zdeseada, ydeseada, xdeseada
z_lr,x_lr,y_lr = 28, 120, 120 # son las dim de physio 

## Testeo
filename = os.listdir(datapath+'/test/')
length = len(filename)

model = torch.load(modelpath_test)
model = nn.DataParallel(model).to(device) # encapsulamiento del modelo en DataParallel

criterion1 = nn.MSELoss()

model.eval()

loss_test = []

print('\n testing...')
for i, data in enumerate(testloader, 0):

    inputs = data[0].reshape(-1,1,z_lr,x_lr,y_lr).to(device)
    labels = data[1].reshape(-1,1,z_hr,x_hr,y_hr).to(device)
    
    #inputs = inputs.float()
    #labels = labels.float()

    print(inputs.type())
    print(labels.type())

    with torch.no_grad():
        outs = model(inputs)

    print(inputs.device)
    print(labels.device)
    print(outs.device)
    
    loss = criterion1(outs, labels)
    loss_test.append(loss.item())

    imagen = "post_modelo_"+str(i) 

    inputs = inputs.reshape(z_lr,x_lr,y_lr)
    labels = labels.reshape(z_hr,x_hr,y_hr)
    outs = outs.reshape(z_hr,x_hr,y_hr)

    inputs_save = inputs.cpu()
    labels_save = labels.cpu()
    outs_save = outs.cpu()

    with h5py.File(os.path.join(savepath, imagen+'.h5'), 'w') as f:
        f.create_dataset('data', data = inputs_save)
        f.create_dataset('target', data = labels_save)
        f.create_dataset('prediction', data = outs_save) 

print("Pérdidas en cada imágen test:")
print(loss_test)

print("Pérdida promedio:")
print(sum(loss_test)/len(loss_test))