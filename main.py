from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import skfuzzy as fuzz
from tensorflow.keras.models import load_model
from keras.models import load_model


# === Cargar modelo y definiciones ===
model = load_model("modelo_cafe") 
# Rango de universos (deben coincidir con los usados en entrenamiento)
acidez_univ = np.arange(4.5, 6.1, 0.1)
cafeina_univ = np.arange(0.8, 2.6, 0.1)
humedad_univ = np.arange(3.0, 12.1, 0.1)
aroma_univ = np.arange(1.0, 10.1, 0.1)

# Reusar funciones de pertenencia
def fuzzy_features(a, c, h, ar):
    from skfuzzy import control as ctrl
    def mf(var, points): return fuzz.trimf(var, points)
    # Miembros difusos
    def val(var, mf_def): return fuzz.interp_membership(var[0], mf_def, var[1])
    return np.array([
        val((acidez_univ, mf(acidez_univ, [4.5, 4.5, 5.0])), a),
        val((acidez_univ, mf(acidez_univ, [4.8, 5.2, 5.6])), a),
        val((acidez_univ, mf(acidez_univ, [5.0, 6.0, 6.0])), a),

        val((cafeina_univ, mf(cafeina_univ, [0.8, 0.8, 1.5])), c),
        val((cafeina_univ, mf(cafeina_univ, [1.2, 1.8, 2.2])), c),
        val((cafeina_univ, mf(cafeina_univ, [1.8, 2.5, 2.5])), c),

        val((humedad_univ, mf(humedad_univ, [3.0, 3.0, 6.0])), h),
        val((humedad_univ, mf(humedad_univ, [4.0, 7.0, 10.0])), h),
        val((humedad_univ, mf(humedad_univ, [8.0, 12.0, 12.0])), h),

        val((aroma_univ, mf(aroma_univ, [1.0, 1.0, 4.0])), ar),
        val((aroma_univ, mf(aroma_univ, [3.0, 5.0, 7.0])), ar),
        val((aroma_univ, mf(aroma_univ, [6.0, 10.0, 10.0])), ar),
    ])

# FastAPI
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Entrada(BaseModel):
    acidez: float
    cafeina: float
    humedad: float
    aroma: float

@app.post("/predecir")
def predecir_entrada(data: Entrada):
    x_orig = np.array([data.acidez, data.cafeina, data.humedad, data.aroma])
    x_fuzzy = fuzzy_features(data.acidez, data.cafeina, data.humedad, data.aroma)
    x_comb = np.concatenate([x_orig, x_fuzzy]).reshape(1, -1)
    pred = model.predict(x_comb)
    clase = np.argmax(pred)
    etiquetas = ['Baja', 'Media', 'Alta']
    return {"calidad": etiquetas[clase], "confianza": float(np.max(pred))}

