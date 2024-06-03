# -*- coding: utf-8 -*-

## Importamos los módulos necesarios 
import numpy as np
import datetime
import time
from modelo import SuperResolutionNet 
import torch.optim as optim 
import os                   
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py  
import matplotlib.pyplot as plt
import numpy as np
import argparse
import wandb 

## Chequeo del uso de varias GPUs
n_gpus = torch.cuda.device_count()
print(n_gpus)
#print(f"Número de GPUs: {n_gpus}")
device = torch.device("cuda")

## Definición de rutas y creación de directorios
datapath  = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/data/h5yentrenamiento/"
#datapath = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/totest/data/h5yentrenamiento2/"
modelpath = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/codigos/entrenamiento/condor/barrido_hiperparámetros/normalizacion"

os.makedirs(modelpath, exist_ok=True)

## Parser 
parser = argparse.ArgumentParser() 
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=8) 
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--weight_decay', type=float, default=0.1) 
parser.add_argument('--wandb_project', type=str)
args = parser.parse_args() 

## Hiperparámetros 
epoch_num = args.num_epochs
epoch_num_char = str(epoch_num)

bs = args.batch_size # batch size
num_workers = args.num_workers

lr = args.lr 
betas= args.betas
weight_decay = args.weight_decay 

lr_update = 0.8 

## Normalización 
max_range = 4000
min_range = 0

# inicialización de una nueva ejecución en wandb
wandb.init(
    # proyecto wandb donde se registrará esta ejecución
    project = args.wandb_project,

    # seguimiento de los hiperparámetros
    config={
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "betas": args.betas,
    "weight_decay": args.weight_decay,
    "epochs": args.num_epochs,
    }
)

## Clase para la obtención de imágenes a alta y baja resolución 
class prepareData_train(Dataset):
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

## Conjunto entrenamiento     
trainset = prepareData_train('train') 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,shuffle=True, num_workers=num_workers, pin_memory=True) # pin_memory=True fuerza la transferencia de datos a CUDA

## Conjunto validación 
validationset = prepareData_train('validation')
validationloader = torch.utils.data.DataLoader(validationset, batch_size=bs,shuffle=True, num_workers=num_workers, pin_memory=True)

## Métodos del entrenamiento - función de pérdida y optimizador 
model = SuperResolutionNet() # nombre de mi red 
device = torch.device("cuda:0")  # primera GPU
model = nn.DataParallel(model).to(device) # encapsulamiento del modelo en DataParallel

criterion1 = nn.L1Loss() # pérdida L1
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

## Pérdidas en el conjunto entrenamiento y prueba 
loss_train_list = [] # contiene el promedio de las pérdidas en cada época 
loss_validation_list = []

## Dim que hemos impuesto en ed_wu_prepare_train_data2
z_hr,x_hr,y_hr = 56, 240, 240 # son zdeseada, ydeseada, xdeseada
z_lr,x_lr,y_lr = 28, 120, 120 # son las dim de physio 

starttime = time.time()

for epoch in range(epoch_num): 
    print('-----------------------------------------------------------------------------')
    print('época'+str(epoch)+':')
    print('-----------------------------------------------------------------------------')
    start_et_train = time.time()  
    model.train()
    loss_batch = []
    print('training...')
    for i, data in enumerate(trainloader, 0):
        start_bt = time.time()

        inputs = data[0].reshape(-1,1,z_lr,x_lr,y_lr).to(device)

        labels_np = np.array(data[1])
        labels = (labels_np - np.min(labels_np)) / (np.max(labels_np) - np.min(labels_np)) * (max_range - min_range) + min_range
        labels = torch.from_numpy(labels_np)
        labels = labels.reshape(-1,1,z_hr,x_hr,y_hr).to(device)

        inputs = inputs.float()
        labels = labels.float()

        outs = model(inputs)
        
        loss = criterion1(outs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_batch.append(loss.item())  

        end_bt = time.time()

        print('tiempo en el batch número '+str(i)+': '+str(end_bt-start_bt)+' s')

        print('epoch:%d - %d, loss:%.10f'%(epoch+1,i+1,loss.item()))
    
    loss_train_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_train_list)

    # log metrics to wandb
    wandb.log({"mean loss epoch (train)": loss_train_list[-1]}) 
    
    end_et_train = time.time()

    print('tiempo de entrenamiento:'+str(end_et_train-start_et_train)+'s')


    model.eval()     # evaluation
    start_et_test = time.time()
    loss_batch = []
    print('\n testing...')

    index = 0
    for i, data in enumerate(validationloader, 0):

        inputs = data[0].reshape(-1,1,z_lr,x_lr,y_lr).to(device)

        labels_np = np.array(data[1])
        labels = (labels_np - np.min(labels_np)) / (np.max(labels_np) - np.min(labels_np)) * (max_range - min_range) + min_range
        labels = torch.from_numpy(labels_np)
        labels = labels.reshape(-1,1,z_hr,x_hr,y_hr).to(device)

        inputs = inputs.float()
        labels = labels.float()
    
        with torch.no_grad():
            outs = model(inputs)

        if index == 0:
                images = []
                imgInput = wandb.Image(inputs[0,0,14,:,:], caption=f"Input")
                images.append(imgInput)
                imgPred = wandb.Image(outs[0,0,28,:,:], caption=f"Pred")
                images.append(imgPred)
                imglabels = wandb.Image(labels[0,0,28,:,:], caption=f"Ref")
                images.append(imglabels)
                wandb.log({"Images": images})
        index += 1
            
        loss = criterion1(outs, labels)
        loss_batch.append(loss.item())

    loss_validation_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_validation_list)
    
    # log metrics to wandb
    wandb.log({"mean loss epoch (validation)": loss_train_list[-1]}) 
    
    end_et_test = time.time() 

    print('tiempo de validación:'+str(end_et_test-start_et_test)+'s')

    print('tiempo total en la época'+str(epoch)+':'+str(end_et_test-start_et_train)+'s')

    if (epoch+1) % 25 == 0:
        torch.save(model, os.path.join(modelpath, 'epoch-%d.pth' % (epoch+1)))
   
    if (epoch+1) % 50 == 0:
        lr = lr*lr_update
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

endtime = time.time()
print('Finished Training. Training time elapsed %.2fs.' %(endtime-starttime))
