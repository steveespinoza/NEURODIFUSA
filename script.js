let model;

window.onload = async () => {
  // Cargar modelo entrenado
  model = await tf.loadGraphModel('./modelo_cafe_tfjs/model.json');
};

// Función para obtener valores actuales
function obtenerValores() {
  return {
    acidez: parseFloat(document.getElementById('acidez').value),
    cafeina: parseFloat(document.getElementById('cafeina').value),
    humedad: parseFloat(document.getElementById('humedad').value),
    aroma: parseFloat(document.getElementById('aroma').value)
  };
}

// Predicción
function predecir() {
  const { acidez, cafeina, humedad, aroma } = obtenerValores();

  // Crear tensor de entrada (original + 12 fuzzy features vacías)
  const input = tf.tensor2d([[acidez, cafeina, humedad, aroma, 
                              0,0,0, 0,0,0, 0,0,0, 0,0,0]]); // (4 + 12) = 16 características

  model.predict(input).array().then(preds => {
    const clase = preds[0].indexOf(Math.max(...preds[0]));
    const etiquetas = ['Baja', 'Media', 'Alta'];
    document.getElementById('resultado').innerText = `☕ Calidad Predicha: ${etiquetas[clase]}`;

    // Mostrar explicación basada en reglas
    const explicacion = evaluarReglas(acidez, cafeina, humedad, aroma);
    document.getElementById('explicacion').innerText = explicacion;
  });
}

// Reglas difusas codificadas en JS
function evaluarReglas(ac, caf, hum, ar) {
  const inRango = (x, min, max) => x >= min && x <= max;

  // Regla 1
  if (inRango(ac, 4.8, 5.2) && inRango(caf, 1.2, 2.2) && inRango(hum, 4.0, 10.0) && ar >= 6.0)
    return "🔍 Activada: Regla 1 – Todos los parámetros ideales → Alta calidad.";

  // Regla 2
  if (inRango(ac, 4.8, 5.2) && inRango(caf, 1.2, 2.2) && inRango(hum, 4.0, 10.0) && inRango(ar, 3.0, 7.0))
    return "🔍 Activada: Regla 2 – Parámetros ideales, pero aroma aceptable → Calidad media.";

  // Regla 3
  if (ac < 4.8 || ac > 5.6 || hum < 4.0 || hum > 10.0 || ar <= 4.0)
    return "🔍 Activada: Regla 3 – Algún parámetro fuera del rango o aroma pobre → Calidad baja.";

  // Regla 4
  if (caf > 1.8 && ar <= 4.0)
    return "🔍 Activada: Regla 4 – Cafeína alta + aroma pobre → Calidad baja.";

  // Regla 5
  if (caf <= 1.2 && ar >= 6.0)
    return "🔍 Activada: Regla 5 – Cafeína baja, pero aroma excelente → Calidad media.";

  return "ℹ️ No se activó ninguna regla específica directamente. Resultado generado por el modelo.";
}
