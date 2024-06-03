# -*- coding: utf-8 -*-
"""
Convierto de dicom 2.0 a dicom 3.0 las imagenes del repositorio FastMRI

Estas imagenes se guardan en la carpeta Clasificadasydicom3

Cada dicom3.0 se llamara FastMRI+contraste+plano+i, donde:
        - contraste = PD o T2
        - plano = AX, COR, SAG
        - i = 1,...,n, donde n es el número de archivos dicom3.0 que hay con el mismo
        contraste+plano 
"""

# importamos los módulos necesarios 
import os
import pydicom
import numpy as np
from dicommanager import DICOMImage 

# definición de funciones 
def dicom2to3(input_path, output_path, min_range, max_range, input_file):
    all_dicom2 = os.listdir(input_path)
    all_dicom2 = [os.path.join(input_path, dicom_file) for dicom_file in os.listdir(input_path) if dicom_file.endswith('.dcm')]
    all_dicom2.sort(key=lambda x: pydicom.dcmread(x).SliceLocation, reverse=True)
    img = []
    for dicom2_file in all_dicom2:
        dicom2_slice = pydicom.dcmread(dicom2_file)
        img.append(dicom2_slice.pixel_array)
    img = np.stack(img)

    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * (max_range - min_range) + min_range

    dicom3_image = DICOMImage()
    
    dicom3_image.meta_data["PixelData"] = img.astype(np.int16).tobytes()
    slices, rows, columns = img.shape
    dicom3_image.meta_data["Columns"] = columns
    dicom3_image.meta_data["Rows"] = rows
    dicom3_image.meta_data["NumberOfSlices"] = slices
    dicom3_image.meta_data["NumberOfFrames"] = slices
    dicom3_image.meta_data["PatientName"] = dicom2_slice.PatientName
    dicom3_image.meta_data["PatientSex"] = dicom2_slice.PatientSex
    dicom3_image.meta_data["StudyID"] = dicom2_slice.StudyID
    dicom3_image.meta_data["PatientID"] =  dicom2_slice.PatientID
    dicom3_image.meta_data["SOPInstanceUID"] = dicom2_slice.SOPInstanceUID
    dicom3_image.meta_data["SeriesDescription"] = dicom2_slice.SeriesDescription
    dicom3_image.meta_data["SeriesNumber"] = dicom2_slice.SeriesNumber
    dicom3_image.meta_data["ImageOrientationPatient"] = dicom2_slice.ImageOrientationPatient
    dicom3_image.meta_data["PixelSpacing"] = dicom2_slice.PixelSpacing
    dicom3_image.meta_data["SliceThickness"] = dicom2_slice.SliceThickness
    dicom3_image.meta_data["SpacingBetweenSlices"] = dicom2_slice.SpacingBetweenSlices
    dicom3_image.meta_data["RepetitionTime"] = dicom2_slice.RepetitionTime
    dicom3_image.meta_data["EchoTime"] = dicom2_slice.EchoTime
    dicom3_image.meta_data["EchoTrainLength"] = dicom2_slice.EchoTrainLength
    dicom3_image.meta_data["StudyDate"] = dicom2_slice.StudyDate
    dicom3_image.meta_data["StudyTime"] = dicom2_slice.StudyTime
    dicom3_image.meta_data["WindowWidth"] = max_range+min_range
    dicom3_image.meta_data["WindowCenter"] = (max_range-min_range)/2
    dicom3_image.meta_data["ProtocolName"] = input_file
    dicom3_image.PixelRepresentation = 0

    dicom3_image.image2Dicom()

    dicom3_image.save(output_path)
    
# definición de parámetros
min_range = 0
max_range = 4000

# definición de las rutas de origen y destino 
#ruta_origen = "C:/Users/USUARIA/Dropbox/BECA CSIC/REPLICA PROYECTO DE REFERENCIA/to test (maria-fastmri)/data/Clasificadas/" # local to test
#ruta_destino = "C:/Users/USUARIA/Dropbox/BECA CSIC/REPLICA PROYECTO DE REFERENCIA/to test (maria-fastmri)/data/Clasificadasydicom3/"
#ruta_origen = "/lhome/ext/i3m121/i3m1214/totest/data/Clasificadas/" # artemisa to test 
#ruta_destino = "/lhome/ext/i3m121/i3m1214/totest/data/Clasificadasydicom3/" # artemisa to test
ruta_origen = "/lhome/ext/i3m121/i3m1214/data/Clasificadas/" # artemisa definitivo
ruta_destino = "/lhome/ext/i3m121/i3m1214/data/Clasificadasydicom3/" # artemisa definitivo

# principal 
contrastes = os.listdir(ruta_origen)
for contraste in contrastes:
    ruta_contraste = ruta_origen + contraste + '/'
    planos = os.listdir(ruta_contraste)
    for plano in planos:
        i=0
        ruta_plano = ruta_contraste + plano + '/'
        pacientes = os.listdir(ruta_plano) 
        for paciente in pacientes:
            ruta_entrada = ruta_plano + paciente + '/'
            i=i+1
            nombredicom = 'FastMRI_'+contraste+'_'+plano+'_'+str(i)
            ruta_salida = ruta_destino + nombredicom 
            os.makedirs(ruta_destino, exist_ok=True)
            dicom2to3(ruta_entrada, ruta_salida, min_range, max_range, nombredicom)
            

