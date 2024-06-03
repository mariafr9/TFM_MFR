# -*- coding: utf-8 -*-
"""
Preparación de los datos de entrenamiento
    1- Homogenización del tamaño de las imágenes de una determinada 
    potenciación y orientación -> interpolación lineal 
    2- Crear la imagen de baja resolución 
    3- Crear un archivo h5 con el pixel array de alta y baja resolución 
    4- clasificación de las imágenes en entrenamiento, test y validación 
"""

# importamos los módulos necesarios 
import numpy as np 
import pydicom as pyd 
from scipy import ndimage
import os 
import glob 
import shutil 
import h5py 
import monai
import sklearn.model_selection

# definición de las rutas 
folder_path = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/totest/data/Clasificadasydicom3"
output_path_h5 = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/totest/data/h5"

# expecificación de la potenciación y orientación 
orientacion = 'ax'
potenciacion = 't2'

# promedio de los tamaños de las imágenes con esa potenciación y orientación 
prom_x = []
prom_y = []
prom_z = []

imagenes = os.listdir(folder_path)   

for imagen in imagenes:
    if orientacion in imagen.lower():
        if potenciacion in imagen.lower():
            ruta_archivo = os.path.join(folder_path, imagen) 
            z,x,y = pyd.dcmread(ruta_archivo).pixel_array.shape 
            prom_x.append(x)
            prom_y.append(y)
            prom_z.append(z) 

# defino el número de pixeles en cada dirección para la homegenización -> dim achieva (36,448,448)
#zdeseada = int(np.mean(prom_z)) # 35 
#xdeseada = int(np.mean(prom_x)) # 305
#ydeseada = int(np.mean(prom_y)) # 305
zdeseada = 56 
xdeseada = 240 
ydeseada = 240 

# definición de parámetros 
min_range = 0
max_range = 4000
down_shape = np.array([28,120,120]) # son las dimensiones de physio 

'''# definición de la función downsampling para crear la imagen a baja resolución
def downSampling(img, min_range, max_range, down_shape):
    kSpace = np.fft.fftshift(np.fft.fftn(img))
    ini_shape = np.array(kSpace.shape)
    center = (ini_shape / 2).astype(int) - (down_shape / 2).astype(int)
    # kSpaceDownSamp = np.zeros(down_shape).astype(complex)

    kSpaceDownSamp = kSpace[center[0]:center[0] + down_shape[0], center[1]:center[1] + down_shape[1],
    center[2]:center[2] + down_shape[2]]
    imgDown = np.abs(np.fft.ifftn((kSpaceDownSamp)))
    img2 = (imgDown - np.min(imgDown)) / (np.max(imgDown) - np.min(imgDown)) * (max_range - min_range) + min_range
    return img2  

# homogenización de las dimensiones, creo imagen a baja resolución, creo el .h5 
for imagen in imagenes:
    if potenciacion in imagen.lower():
        if orientacion in imagen.lower():
            # homogenización del tamaño de las imágenes 
            path = os.path.join(folder_path, imagen)
            dicom = pyd.dcmread(path) 
            z,x,y = dicom.pixel_array.shape 
            print(dicom.pixel_array.shape)
            factor = (zdeseada/z,xdeseada/x,ydeseada/y) 
            interpolada = ndimage.zoom(dicom.pixel_array, factor) 
            print(interpolada.shape)
            print('------------------')
            # creo la imagen a baja resolución y con ruido 
            baja_resolucion = downSampling(interpolada, min_range, max_range, down_shape) 
            noise = monai.transforms.RandRicianNoise(prob=1.0, std=0.4, relative=True, channel_wise=True, sample_std=False)
            noise_seed = np.random.randint(10000) 
            noise.set_random_state(noise_seed)
            ruidoybajaresol = noise(baja_resolucion) 
            # creo el archivo .h5
            os.makedirs(output_path_h5, exist_ok=True) 
            with h5py.File(os.path.join(output_path_h5, imagen+'.h5'), 'w') as f:
                f.create_dataset('data', data=ruidoybajaresol)
                f.create_dataset('target', data=interpolada) '''
                
train_test_ratio = 0.3
validation_test_ratio = 0.5 

input_path = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/data/data_h5"
output_path = "/lhome/ext/i3m121/i3m1214/replica_ed_wu/data/h5yentrenamiento"
seed = 100

def main(train_test_ratio, validation_test_ratio, input_path, output_path):
    data_flist = glob.glob(os.path.join(input_path, '*.h5'))
    rng = np.random.default_rng(seed) 
    rng.shuffle(data_flist)
    train, _test = \
        sklearn.model_selection.train_test_split(                          
        data_flist, test_size=train_test_ratio, random_state=seed
    ) 
    test, val = \
        sklearn.model_selection.train_test_split(
        _test, test_size=validation_test_ratio, random_state=seed
    )
    
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    for ruta in train:
        fragmentos = ruta.split("/")
        file = fragmentos[-1] 
        shutil.move(
            input_path + '/' + file, 
            os.path.join(output_path, 'train') + '/' + file 
        )
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    for ruta in test:
        fragmentos = ruta.split("/")
        file = fragmentos[-1] 
        shutil.move(
            input_path + '/' + file,
            os.path.join(output_path, 'test') + '/' + file
        )
    os.makedirs(os.path.join(output_path, 'validation'), exist_ok=True)
    for ruta in val:
        fragmentos = ruta.split("/")
        file = fragmentos[-1] 
        shutil.move(
            input_path + '/' + file,
            os.path.join(output_path, 'validation') + '/' + file
        )


main(train_test_ratio,validation_test_ratio,input_path,output_path)
