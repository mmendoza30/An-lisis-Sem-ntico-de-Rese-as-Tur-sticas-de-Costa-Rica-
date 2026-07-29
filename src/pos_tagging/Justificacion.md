Después de probar ambas herramientas, elegimos spaCy como la herramienta principal para este proyecto. 

Las razones son:
- Mayor precisión en español: spaCy fue entrenado específicamente con textos en español. NLTK, en cambio, requiere un "tagger" propio (nosotros lo entrenamos con el corpus `cess_esp`) y aun así da errores y etiquetas "None".

- Manejo de la ambigüedad: Como vimos en el ejemplo de "bajo", spaCy entiende mejor el contexto. Esto es clave en reseñas donde las palabras pueden tener varios significados.

- Etiquetas más simples: spaCy usa Universal POS (12 etiquetas fáciles: NOUN, VERB, ADJ, etc.). NLTK usa un sistema muy detallado (más de 50 etiquetas) que es complicado de interpretar y no nos aporta valor extra para nuestro análisis (comparar reseñas positivas vs negativas).

- Velocidad y estabilidad: spaCy procesa cientos de reseñas mucho más rápido y no se queda sin etiquetar palabras.

Por todo esto, spaCy es la mejor opción para proyectos de análisis de texto en español como el nuestro. 
NLTK sirve como una herramienta para entender las dificultades del etiquetado, pero no la recomendamos para aplicaciones reales.
