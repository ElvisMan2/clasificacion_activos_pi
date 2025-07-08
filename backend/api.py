from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Cargar modelo y vectorizador
modelo = joblib.load("./v1.1/modelo_svm.pkl")
vectorizer = joblib.load("./v1.1/vectorizer_tfidf.pkl")

# Instancia FastAPI
app = FastAPI(title="Clasificación de Activos de PI")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Esquema de entrada
class DescripcionInput(BaseModel):
    descripcion: str

@app.post("/predict")
def predecir_categoria(data: DescripcionInput):
    # Transformar el texto 
    descripcion_tfidf = vectorizer.transform([data.descripcion])
    # Predecir la categoría
    prediccion = modelo.predict(descripcion_tfidf)
    return {"categoria_predicha": prediccion[0]}


@app.get("/", response_class=HTMLResponse)
def serve_form():
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()