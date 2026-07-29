Para evaluar qué herramienta funciona mejor en español, probamos una frase con una palabra ambigua:  
**"El bajo precio del hotel es bajo comparado con otros."**  
La palabra "bajo" puede ser adjetivo (cuando describe algo, ej. "precio bajo") o preposición (ej. "bajo la mesa").  

Resultados obtenidos:

| Token     | Etiqueta NLTK       | Etiqueta spaCy      |
|-----------|-------------------- |---------------------|
| El        | da0ms0 (artículo)   | DET (determinante)  |
| **bajo**  | sps00 (preposición) | ADJ (adjetivo)      |
| precio    | ncms000 (sustantivo)| NOUN                |
| del       | spcms (contracción) | ADP                 |
| hotel     | ncms000             | NOUN                |
| es        | vsip3s0 (verbo)     | AUX                 |
| **bajo**  | sps00 (preposición) | ADP (preposición)   |
| comparado | aq0msp (adjetivo)   | ADJ                 |
| con       | sps00 (preposición) | ADP                 |
| otros     | di0mp0 (pronombre)  | PRON                |
 
- NLTK etiquetó ambos "bajo" como preposición, aunque el primero es adjetivo. Esto pasa porque NLTK no mira el contexto solo una la palabra suelta.  
- spaCy acertó en el primer "bajo" (lo reconoció como adjetivo) y falló en el segundo. Aunque no es perfecto, spaCy es más consciente del contexto.

Además, en una reseña real:
vimos que NLTK dejó palabras sin etiqueta (None), como en `('pésima', None)`. Esto no ocurrió con spaCy.  
Esto indica que el modelo de NLTK es menos completo para el español.

**Conclusión:**  
spaCy es más preciso y robusto que NLTK para etiquetar español, especialmente en palabras ambiguas y texto informal como reseñas con las que trabajamos.