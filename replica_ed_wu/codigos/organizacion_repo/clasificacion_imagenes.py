# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:34:40 2024
@author: María Fernández Ramos 

Organización de las imágenes obtenidas del repositorio FastRMI para el 
entrenamiento de redes neuronales

Se clasifican en primer lugar en función del tipo de contraste; T2 o PD, y 
dentro de esta clasificación, en plano SAGITAL, CORONAL o AXIAL
"""

# importamos los módulos necesarios 
import os 
import shutil 
import pydicom as pyd 

# definición de las ubicaciones de origen y de destino -> esta para las imágenes test
ruta_origen = "/lhome/ext/i3m121/i3m1214/data/FastMRI/test/knee_mri_clinical_seq_batch2" #artemisa
ruta_destino = "/lhome/ext/i3m121/i3m1214/data/Clasificadas/test" #artemisa 
#ruta_origen = "C:/Users/USUARIA/Dropbox/BECA CSIC/REPLICA PROYECTO DE REFERENCIA/dicom" #local
#ruta_destino = "C:/Users/USUARIA/Dropbox/BECA CSIC/REPLICA PROYECTO DE REFERENCIA/clasificadas" #local

# clasificación en contrastes t2 y pd 
contraste_t2 = os.path.join(ruta_destino,"T2")
contraste_pd = os.path.join(ruta_destino,"PD") 

# dentro de cada contraste, en sagital, axial y coronal 
planos = ["SAG","AX","COR"] 
for plano in planos:
    rutat2_plano = os.path.join(contraste_t2,plano) 
    os.makedirs(rutat2_plano,exist_ok=True) 
    rutapd_plano = os.path.join(contraste_pd,plano) 
    os.makedirs(rutapd_plano,exist_ok=True)  

# obtengo una lista de las carpetas de ruta_origen 
carpetas_ruta_origen = os.listdir(ruta_origen)  

# clasificación 
for carpeta in carpetas_ruta_origen:
    ruta_carpeta = os.path.join(ruta_origen, carpeta)
    estudios = os.listdir(ruta_carpeta)   
    for estudio in estudios:
        ruta_pacientes = os.path.join(ruta_carpeta, estudio)
        pacientes = os.listdir(ruta_pacientes) 
        for paciente in pacientes:
            ruta_paciente = os.path.join(ruta_pacientes, paciente) 
            archivos = os.listdir(ruta_paciente) 
            image = pyd.dcmread(os.path.join(ruta_paciente,archivos[0])) 
            if 't2' in image.SeriesDescription.lower():
                if 'sag' in image.SeriesDescription.lower(): 
                    ruta_destino_contrasteyplano = os.path.join(contraste_t2,"SAG")
                    ruta_destino_paciente = os.path.join(ruta_destino_contrasteyplano,paciente) 
                    os.makedirs(ruta_destino_paciente,exist_ok=True)  
                    for archivo in archivos:
                        shutil.copy(os.path.join(ruta_paciente, archivo), ruta_destino_paciente) 
                if 'cor' in image.SeriesDescription.lower(): 
                    ruta_destino_contrasteyplano = os.path.join(contraste_t2,"COR")
                    ruta_destino_paciente = os.path.join(ruta_destino_contrasteyplano,paciente) 
                    os.makedirs(ruta_destino_paciente,exist_ok=True)  
                    for archivo in archivos:
                        shutil.copy(os.path.join(ruta_paciente, archivo), ruta_destino_paciente) 
                if 'ax' in image.SeriesDescription.lower(): 
                    ruta_destino_contrasteyplano = os.path.join(contraste_t2,"AX")
                    ruta_destino_paciente = os.path.join(ruta_destino_contrasteyplano,paciente) 
                    os.makedirs(ruta_destino_paciente,exist_ok=True)  
                    for archivo in archivos:
                        shutil.copy(os.path.join(ruta_paciente, archivo), ruta_destino_paciente) 
            if 'pd' in image.SeriesDescription.lower():
                if 'sag' in image.SeriesDescription.lower():
                    ruta_destino_contrasteyplano = os.path.join(contraste_pd,"SAG")
                    ruta_destino_paciente = os.path.join(ruta_destino_contrasteyplano,paciente) 
                    os.makedirs(ruta_destino_paciente,exist_ok=True)  
                    for archivo in archivos:
                        shutil.copy(os.path.join(ruta_paciente, archivo), ruta_destino_paciente) 
                if 'cor' in image.SeriesDescription.lower(): 
                    ruta_destino_contrasteyplano = os.path.join(contraste_pd,"COR")
                    ruta_destino_paciente = os.path.join(ruta_destino_contrasteyplano,paciente) 
                    os.makedirs(ruta_destino_paciente,exist_ok=True)  
                    for archivo in archivos:
                        shutil.copy(os.path.join(ruta_paciente, archivo), ruta_destino_paciente) 
                if 'ax' in image.SeriesDescription.lower(): 
                    ruta_destino_contrasteyplano = os.path.join(contraste_pd,"AX")
                    ruta_destino_paciente = os.path.join(ruta_destino_contrasteyplano,paciente) 
                    os.makedirs(ruta_destino_paciente,exist_ok=True)  
                    for archivo in archivos:
                        shutil.copy(os.path.join(ruta_paciente, archivo), ruta_destino_paciente) 
            

            
            
            
            
            
            
            
            
            
            
            
            
            
        