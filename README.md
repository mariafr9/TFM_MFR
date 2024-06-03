# Redes neuronales para resonancia magnética a bajo campo
Este repositorio contiene el trabajo realizado por **María Fernández Ramos** para replicar el estudio descrito en el artículo ["Deep learning enabled fast 3D brain MRI at 0.055 tesla"](https://www.science.org/doi/10.1126/sciadv.adi9327), aplicándolo al caso particular de imágenes de rodilla. El objetivo principal es mejorar la calidad de las imágenes obtenidas por el escáner de resonancia magnética (RM) de bajo campo del MRILab.


El objetivo es dar los primeros pasos hacia la mejora de la calidad de las imágenes obtenidas por el escáner de RM de bajo campo del MRILab.

## Organización del código:
A continuación, se muestra un esquema con la estructura de carpetas de este repositorio:
-	codigos
    - entrenamiento
        - condor
            - barrido_hiperparámetros 
            - batch_size
            - learning_rate 
            - learning_rate2 
            - weight_decay
            - definitivo1
        - organización_repo 
        - test
- data
    - h5yentrenamiento
        - test
        - train 
        - validation
<br><br>

## Principales códigos de ejecución:
- **clasificación_imagenes.py:** Organización de las imágenes obtenidas del repositorio *FastRMI* para el entrenamiento de redes neuronales.Se clasifican en primer lugar en función del tipo de contraste; T2/PD, y dentro de esta clasificación, en plano SAGITAL, CORONAL o AXIAL.

- **dicommanager.py:** Código creado por *Teresa Guallart-Naval*. Contiene funciones auxiliares para la generación de un archivo `DICOM 3.0`.

- **todicom3.py:** Genera los archivos `DICOM 3.0` a partir de los archivos `DICOM 2.0`.

- **prepare_train_data.py:** Preparación de los datos de entrenamiento. 1- Homogenización del tamaño de las imágenes de una determinada contraste y orientación mediante interpolación trilineal. 2- Crear la imagen de baja resolución. 3- Crear un archivo h5 con las matrices de píxeles de las matrices de alta y baja resolución. 4- Clasificación de las imágenes en entrenamiento, test y validación.

- **modelo.py:** Arquitectura de la red.

- **totrain.py:** Código de entrenamiento.

- **test_fastMRI.py:** Aplicar el modelo entrenado a las imágenes test.


## Información adicional 📖

Puedes encontrar mucho más sobre este proyecto en mi TFM del Máster en Física Biomédica por la [Universidad Complutense de Madrid](https://https://www.ucm.es/).

## Construido con 🛠️
* [Python 3.7.9](https://www.python.org/downloads/release/python-379/) - Compilador


## Autores ✒️
* **María Fernandez** - *Trabajo Inicial y Documentación* - [MariaFR9](https://github.com/mariafr9)
* **Teresa Guallart-Naval** - *Principal colaboradora*
* **Joseba Alonso Otamendi** - *Supervisión y revisión de la documentación* -
* **José Miguel Algarín Guisado** - *Supervisión y revisión de la documentación*
