import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import streamlit as st

def leer_dato(uploaded_file):
    dato = pickle.loads(uploaded_file.getvalue())
    dato = dato.reshape(dato.shape[0]* dato.shape[1])  

    return dato

# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Detección de anomalías cardiacas", layout="wide")
st.title("Detección de anomalías cardiacas con autoencoders")

c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%

with c30:
    uploaded_file = st.file_uploader(
        "", type = 'pkl',
        key="1",
    )

    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        info_box_wait = st.info(
            f"""
                Realizando la clasificación...
                """)

        # Acá viene la predicción con el modelo
        dato = leer_dato(uploaded_file)
        
        model_cargado = joblib.load("modelo_entrenado.pkl")
        
        # rl = Autoencoder()
        # autoencoder = cargar_modelo_preentrenado('autoencoder.pth')
        # prediccion = predecir(autoencoder, dato, UMBRAL)
        prediccion=model_cargado.predict(dato.reshape(1, -1))
        # categoria = obtener_categoria(prediccion)

        # Y mostrar el resultado
        info_box_result = st.info(f"""
        	El dato analizado corresponde a un sujeto: {prediccion[0]}
        	""")

    else:
        st.info(
            f"""
                👆 Debe cargar primero un dato con extensión .pkl
                """
        )

        st.stop()