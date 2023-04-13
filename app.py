import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import streamlit as st
from PIL import Image

def leer_dato(image):  
    img_array = np.asarray(image.convert('L').resize((28, 28)))
    img_array = 1 - (img_array / 255)
    img_array = img_array.reshape(1, 784)

    return img_array

# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Detección de numeros con mnist", layout="wide")
st.title("Detección de numeros con mnist")

# c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%

# with c30:
uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:        
    image = Image.open(uploaded_file) 
    # Acá viene la predicción con el modelo
    dato = leer_dato(image)  
    st.image(image, caption='Uploaded Image', width=300)
    model_cargado = joblib.load("modelo_entrenado.pkl")
    
    prediccion=model_cargado.predict(dato)
    # Y mostrar el resultado
    info_box_result = st.info(f"""Numero predecido con el modelo completo: {prediccion[0]}""")

    model_cargado2 = joblib.load("modelo_reducido_0_8.pkl")    
    model_pca = joblib.load("pca_2_componentes.pkl")

    dato=model_pca.transform(dato)

    prediccion=model_cargado2.predict(np.real(dato))
    # Y mostrar el resultado
    info_box_result = st.info(f"""Numero predecido con el modelo reducido 0 - 8: {prediccion[0]}""")
    st.stop()