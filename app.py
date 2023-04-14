import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize

def leer_dato2(img_array):  
    # img_array = np.asarray(image.convert('L').resize((28, 28)))
    img_array = 1 - (img_array / 255)
    img_array = img_array.reshape(1, 784)

    return img_array

def leer_dato(image):  
    img_array = np.asarray(image.convert('L').resize((28, 28)))
    img_array = 1 - (img_array / 255)
    img_array = img_array.reshape(1, 784)

    return img_array

def predecir(modelo,dato):
    return modelo.predict(dato)

# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Detección de numeros con mnist", layout="wide")
st.title("Detección de numeros con mnist")
prediccion=None
uploaded_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

# Creamos una lista de opciones para los botones de opción
opciones = ["Modelo completo, todos los numeros, sin reducción de dimensionalidad",
            "Modelo solo con números 0 - 8, reducido con PCA"
            ,            "Modelo solo con números 0 - 8, reducido con SVD"
            ]
# Creamos una variable para almacenar el valor seleccionado
opcion_seleccionada = st.radio("Seleccione una opción:", opciones)

if uploaded_file is not None:        
    image = Image.open(uploaded_file) 
    # Acá viene la predicción con el modelo
    dato = leer_dato(image)  

    if opcion_seleccionada=="Modelo completo, todos los numeros, sin reducción de dimensionalidad":
        prediccion=predecir(joblib.load("modelo_entrenado.pkl"),dato)

    if opcion_seleccionada=="Modelo solo con números 0 - 8, reducido con PCA":
        model_pca = joblib.load("pca_2_componentes.pkl")
        dato=model_pca.transform(dato)
        prediccion=predecir(joblib.load("modelo_reducido_0_8.pkl"),np.real(dato))
        
    if opcion_seleccionada=="Modelo solo con números 0 - 8, reducido con SVD":
        model_svd = joblib.load("svd_2_componentes.pkl")
        dato=model_svd.transform(dato)
        prediccion=predecir(joblib.load("modelo_reducido_0_8_svd.pkl"),np.real(dato))

    st.image(image, caption=prediccion[0], width=300)


st.write("Opción seleccionada:", opcion_seleccionada)
if prediccion is not None:
    st.success(f"### El numero detectado es {prediccion[0]} ")

#----------------------------------------- probando



st.write("proyecto para streamlit premium")
canvas_result = st_canvas(
    fill_color="#FFF",
    stroke_width=10,
    stroke_color="#000",
    background_color="#FFF",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Obtener el resultado del widget de dibujo
if canvas_result.image_data is not None:
    
    img1=resize(canvas_result.image_data,(28,28))
    st.write((img1).shape)
    # # img = img.resize((28, 28))
    # img = ImageOps.invert(img)
    img = np.array(img1[:,:,0:1]) / 255.0
    img = img.reshape(1, 784)

    prediccion=predecir(joblib.load("modelo_entrenado.pkl"),(img))
    st.image(np.array(img1[:,:,0:1]) , caption=prediccion[0])
st.stop()
