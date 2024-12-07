from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import onnxruntime as ort
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from fastapi.responses import PlainTextResponse
import base64
from io import BytesIO

# Constantes
FORMATOS_VALIDOS = {'.jpg', '.jpeg', '.png'}
FACTOR_ESCALA = 0.005999880002399953

# Inicializar FastAPI
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8081",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo ONNX, scaler y encoder
ort_session = ort.InferenceSession("modelo_papas.onnx")
with open('scaler_papas.pkl', 'rb') as f:
    scaler = pickle.load(f)

encoder = OrdinalEncoder()
datos = pd.read_csv('dataset_caracteristicas_papas.csv',
                    delimiter=',', header=0)
encoder.fit(datos[['Especie']])

# Función para extraer características


def extraer_caracteristicas(imagen, especie):
    try:
        # Asegúrate de que la imagen esté en formato adecuado
        if imagen is None:
            print("Advertencia: La imagen no se pudo cargar.")
            return None

        # Conversión a escala de grises y espacio HSV
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        # Segmentación para encontrar contornos
        _, umbral = cv2.threshold(
            gris, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(
            umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return None

        # Selección del contorno más grande
        contorno = max(contornos, key=cv2.contourArea)
        area_px = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        # Conversión de área (aproximada) a cm^2
        escala_cm2 = (FACTOR_ESCALA) ** 2
        area_cm2 = area_px * escala_cm2
        circularidad = (4 * np.pi * area_px) / (perimetro ** 2)

        # Relación de aspecto
        x, y, w, h = cv2.boundingRect(contorno)
        relacion_aspecto = float(w) / h

        # Color promedio en espacio HSV
        color_promedio = cv2.mean(hsv)[:3]

        # Cálculo de GLCM y propiedades
        glcm = graycomatrix(gris, [1], [0, np.pi / 2],
                            symmetric=True, normed=True, levels=256)
        contraste = graycoprops(glcm, 'contrast').mean()
        homogeneidad = graycoprops(glcm, 'homogeneity').mean()
        entropia = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
        brillo_promedio = color_promedio[2]

        return [
            f"V{str(especie).zfill(2)}", area_cm2, circularidad,
            relacion_aspecto, *color_promedio, contraste, entropia,
            homogeneidad, brillo_promedio
        ]
    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return None


# Ruta de predicción


# Ruta de predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validar el formato del archivo
        extension = os.path.splitext(file.filename)[-1].lower()
        if extension not in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            raise HTTPException(
                status_code=400, detail="Formato de archivo no permitido.")

        # Leer el archivo cargado
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        print(f"Imagen recibida: {image}")

        if image is None:
            raise HTTPException(
                status_code=400, detail="No se pudo leer la imagen.")

        # Extraer características
        vector = extraer_caracteristicas(image, especie=1)
        if vector is None:
            raise HTTPException(
                status_code=400, detail="No se pudieron extraer características de la imagen.")

        # Realizar predicción
        datos_entrada = np.array([vector[1:]])  # Ignorar la columna "Especie"
        datos_entrada_estandarizados = scaler.transform(datos_entrada)

        input_name = ort_session.get_inputs()[0].name
        result = ort_session.run(
            None, {input_name: datos_entrada_estandarizados.astype(np.float32)})

        prediccion = result[0]
        especie_predicha = encoder.inverse_transform(prediccion.reshape(-1, 1))

        return {"especie": especie_predicha[0][0]}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error procesando la predicción: {str(e)}")
