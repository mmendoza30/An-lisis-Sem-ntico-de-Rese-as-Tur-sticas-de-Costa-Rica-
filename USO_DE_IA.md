# Documentación sobre el Uso de Inteligencia Artificial

## 1. Herramientas Utilizadas y Tareas Específicas

Durante el desarrollo de este proyecto se emplearon herramientas de Inteligencia Artificial Generativa como apoyo en diversas fases del flujo de trabajo:

* **Claude / ChatGPT / Gemini:**
  * **Explicación e interpretación teórica:** Comprensión profunda de las diferencias entre las arquitecturas **CBOW** y **Skip-Gram** de Word2Vec, así como el análisis de embeddings contextuales en **BETO**.
  * **Redacción y depuración de código:** Asistencia en la estructuración de funciones para *mean pooling* (vectores promedio por reseña), visualizaciones t-SNE, matrices de confusión y modelos de clasificación (scikit-learn).
  * **Optimización y formateo de documentación:** Revisión de estilo y corrección ortográfica de las interpretaciones para los notebooks de Jupyter.

---

## 2. Ejemplos de Prompts Representativos

A continuación se presentan 4 ejemplos de prompts utilizados durante el desarrollo del proyecto:

### Prompt 1: Interpretación de Vectores y Comparativa de Arquitecturas
> *¿Cómo puedo interpretar esta diferencia entre CBOW y Skip-Gram de forma no tan tecnica?"*

### Prompt 2: Creación de Función para Promedio Vectorial
> *"Necesito una función en Python para obtener el vector promedio (mean pooling) de una reseña usando un modelo Word2Vec de gensim. Si una palabra no está en el vocabulario del modelo, debe ignorarla. Muéstrame cómo aplicarlo a una columna de un DataFrame de pandas."*



### Prompt 4: Solución de Errores de Análisis / Entorno
> *"Me sale el error `<versionspec>, RequirementsTokenType... expected, got 'DE'` al intentar parsear un archivo en Python. ¿A qué se debe este error de sintaxis y cómo lo corrijo?"*

---

## 3. Reflexión sobre el Aprendizaje con IA


---

## 4. Modificaciones y Validaciones al Código/Análisis Generado

Aunque la IA proporcionó plantillas iniciales y sugerencias de código/texto, se realizaron ajustes clave para garantizar la validez del proyecto:

1. **Ajuste de Interpretaciones:** Se editaron y contextualizaron los textos generados por la IA para alinearlos estrictamente con las particularidades del dataset costarricense de hoteles, parques y restaurantes