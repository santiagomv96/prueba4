import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import streamlit as st
from PIL import Image


def leer_dato(image):
    # dato = pickle.loads(uploaded_file.getvalue())
    # dato = dato.reshape(dato.shape[0]* dato.shape[1])      
    img_array = np.asarray(image.convert('L').resize((28, 28)))
    img_array = 1 - (img_array / 255)
    img_array = img_array.reshape(1, 784)

    return img_array

# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Detección de numeros con mnist", layout="wide")
st.title("Detección de numeros con mnist")

c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%

with c30:
    # uploaded_file = st.file_uploader("", type = 'pkl',key="1",)
    uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:        
        image = Image.open(uploaded_file) 
        #mostrar img cargada
        # file_container = st.expander("Verifique el archivo .pkl que acaba de subir")
        # info_box_wait = st.info(f"""Realizando la clasificación...""")

        # Acá viene la predicción con el modelo
        dato = leer_dato(image)  
        # image = image.resize((28, 28), resample=Image.BILINEAR)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        model_cargado = joblib.load("modelo_entrenado.pkl")
        
        # rl = Autoencoder()
        # autoencoder = cargar_modelo_preentrenado('autoencoder.pth')
        # prediccion = predecir(autoencoder, dato, UMBRAL)
        prediccion=model_cargado.predict(dato)
        # categoria = obtener_categoria(prediccion)

        # Y mostrar el resultado
        info_box_result = st.info(f"""El dato analizado corresponde a un sujeto: {prediccion[0]}""")

    # else:
    #     st.info(f"""👆 Debe cargar primero un dato con extensión .pkl""")

        st.stop()