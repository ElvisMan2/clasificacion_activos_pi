from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo y vectorizador
modelo = joblib.load("modelo_svm.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")

# Instancia FastAPI
app = FastAPI(title="Clasificación de Activos de PI")

# Esquema de entrada
class DescripcionInput(BaseModel):
    descripcion: str

@app.post("/predict")
def predecir_categoria(data: DescripcionInput):
    # Transformar el texto usando el TF-IDF
    descripcion_tfidf = vectorizer.transform([data.descripcion])
    # Predecir la categoría
    prediccion = modelo.predict(descripcion_tfidf)
    return {"categoria_predicha": prediccion[0]}
