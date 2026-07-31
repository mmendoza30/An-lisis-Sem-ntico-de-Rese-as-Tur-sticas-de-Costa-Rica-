# Uso de Inteligencia Artificial en el Desarrollo del Proyecto Reseñas Turisticas de Costa Rica

**Autores:** Mendoza Morales Mónica · Contreras Artavia Fernando
**Curso:** Minería de textos — BD-163 · Colegio Universitario de Cartago · 2026

---
## 1. Qué herramientas de IA se utilizaron y para qué tareas específicas 

Durante el desarrollo del proyecto Análisis Semántico de Reseñas Turísticas de Costa Rica, se utilizaron diferentes asistentes de inteligencia artificial como apoyo para acelerar el desarrollo del código, resolver problemas técnicos, mejorar la organización del proyecto y comprender el funcionamiento de las técnicas de Procesamiento de Lenguaje Natural (PLN). Las decisiones finales, integración y validación fueron realizadas por los integrantes del proyecto.
**Herramientas utilizadas:** Gemini, deepseek, chatgpt, Claude

---

### Web Scraping

La IA apoyó en la elaboración del código base para realizar Web Scraping de reseñas turísticas respetando las políticas de acceso mediante User-Agent, tiempos de espera y manejo de errores.
**Aporte:** * Automatización de la recolección de reseñas. * Manejo de excepciones. * Limpieza inicial de la información obtenida.

---

### MongoDB

Se utilizó IA para comprender la estructura documental más adecuada para almacenar las reseñas del Proyecto 1 junto con las nuevas reseñas obtenidas mediante scraping.
**Aporte:** * Organización de la colección. * Definición de campos del documento. * Consultas básicas para validar la información almacenada.
---

### Preprocesamiento del corpus

La IA apoyó durante el desarrollo del proceso de limpieza del corpus, selección de columnas, normalización de fechas, traducción de reseñas en inglés al español, tokenización mediante spaCy y NLTK, y generación de etiquetas gramaticales (POS Tags).
**Aporte:** * Optimización del pipeline de limpieza. * Resolución de errores relacionados con codificación, rutas y formatos de fechas. * Automatización de la tokenización y etiquetado gramatical.
---

### Implementación de BETO

La IA fue utilizada como apoyo para comprender el funcionamiento del modelo BETO de Hugging Face y desarrollar el notebook correspondiente.

Se implementaron las siguientes funcionalidades:

* Generación de embeddings para cada reseña.
* Análisis de polisemia contextual.
* Búsqueda semántica mediante similitud coseno.
* Predicción de palabras utilizando Masked Language Model (MLM).
* 
**Aporte:** * Comprensión del funcionamiento interno de BETO. * Implementación de embeddings utilizando el promedio de los tokens. * Desarrollo de ejemplos prácticos sobre polisemia y búsqueda semántica.

## 2. Ejemplos de prompts utilizados  

---
“¿Cómo puedo almacenar un corpus de reseñas turísticas en MongoDB utilizando un esquema documental?”
---
“Ayúdame a limpiar un corpus de reseñas manteniendo únicamente determinadas columnas.”
---
¿Cómo puedo conservar únicamente las reseñas en español y traducir una muestra de reseñas en inglés?
---
“Explícame cómo generar embeddings utilizando el modelo BETO de Hugging Face.”
---
“¿Cómo implementar Word2Vec mediante CBOW y Skip-Gram?”
---
“Ayúdame a corregir errores relacionados con pandas, spaCy y Transformers.”
---


## 3. Reflexión sobre cómo la IA ayudó en el aprendizaje  

---
La utilización de herramientas de inteligencia artificial permitió comprender con mayor profundidad el desarrollo completo de un proyecto de minería de textos.

La IA sirvió como apoyo para entender el funcionamiento de modelos modernos de lenguaje como BETO, la generación de embeddings, la tokenización mediante spaCy y NLTK, el almacenamiento documental en MongoDB y la comparación entre diferentes representaciones del texto como Bag of Words, TF-IDF y Word2Vec.

Asimismo, facilitó la resolución de problemas técnicos durante el desarrollo, permitiendo dedicar más tiempo al análisis de resultados y a la interpretación de los modelos utilizados. También ayudó en la organización del proyecto, la elaboración de documentación técnica y la preparación del dashboard final.

El uso de estas herramientas fortaleció el aprendizaje práctico de técnicas de Procesamiento de Lenguaje Natural y permitió comprender mejor cómo integrar distintas tecnologías dentro de un mismo pipeline de análisis semántico.
---

## 4. Qué modificaciones se hicieron al código/análisis generado por IA  

---
Adaptación del proceso de limpieza para conservar únicamente las columnas requeridas del corpus.
Traducción y selección de las reseñas en inglés antes de integrarlas al corpus final.
Ajuste del pipeline de tokenización utilizando spaCy y un etiquetador de NLTK entrenado con el corpus cess_esp.
Corrección de rutas de archivos para trabajar con la estructura definitiva del proyecto.
Adaptación del modelo BETO al corpus de reseñas turísticas utilizando el modelo oficial dccuchile/bert-base-spanish-wwm-cased.
Implementación de la generación de embeddings mediante el promedio de los tokens.
Desarrollo del análisis de polisemia utilizando similitud coseno entre diferentes contextos.
Construcción de un buscador semántico basado en embeddings generados por BETO.
Implementación de ejemplos de Masked Language Model utilizando frases del dominio turístico.
Integración de MongoDB con el corpus del Proyecto 1 y las nuevas reseñas obtenidas mediante Web Scraping.
Ajustes al dashboard para representar correctamente los resultados obtenidos durante el análisis.

Estas modificaciones permitieron adaptar las sugerencias generadas por la IA al dominio específico del proyecto y garantizar que el código final fuera funcional, organizado y coherente con los objetivos establecidos por el curso.
---