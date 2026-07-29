"""
Módulo de comparación entre NLTK (Penn Treebank en español) y spaCy (Universal POS).
"""

import pandas as pd
import spacy
import nltk
from nltk.corpus import cess_esp
from pathlib import Path
import ast


def cargar_datos_pos(ruta_csv: str):
    """Carga el CSV con las columnas tokens_spacy y tokens_nltk"""
    df = pd.read_csv(ruta_csv)
    df['tokens_spacy'] = df['tokens_spacy'].apply(ast.literal_eval)
    df['tokens_nltk'] = df['tokens_nltk'].apply(ast.literal_eval)
    return df


def comparar_etiquetas_en_frase(frase: str, tagger_nltk, nlp_spacy):
    """Toma una frase, aplica ambos taggers y devuelve una tabla comparativa"""
    tokens_nltk = nltk.word_tokenize(frase, language='spanish')
    tagged_nltk = tagger_nltk.tag(tokens_nltk)
    doc = nlp_spacy(frase)
    comparacion = []
    for token_spacy in doc:
        token_text = token_spacy.text
        etq_nltk = next((tag for t, tag in tagged_nltk if t == token_text), '—')
        comparacion.append({
            'Token': token_text,
            'Etiqueta NLTK (español)': etq_nltk,
            'Etiqueta spaCy (Universal)': token_spacy.pos_,
        })
    return pd.DataFrame(comparacion)


def encontrar_ruta_csv():
    """Busca la carpeta 'data/processed/resenas_pos_tagged.csv'"""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # subir hasta 10 niveles
        candidate = current / 'data' / 'processed' / 'resenas_pos_tagged.csv'
        if candidate.exists():
            return candidate
        current = current.parent
    return None


if __name__ == "__main__":
    # 1. Cargar tagger NLTK
    print("Cargando tagger NLTK entrenado con cess_esp...")
    entrenamiento = cess_esp.tagged_sents()
    tagger_nltk = nltk.UnigramTagger(entrenamiento)

    # 2. Cargar modelo spaCy
    print("Cargando spaCy modelo es_core_news_md...")
    nlp = spacy.load('es_core_news_md')

    # 3. Buscar el CSV
    ruta_csv = encontrar_ruta_csv()
    if ruta_csv:
        print(f" CSV encontrado en: {ruta_csv}")
        df = cargar_datos_pos(str(ruta_csv))
        print(f"Dataset cargado con {len(df)} reseñas.")
    else:
        print(" No se encontró el archivo 'resenas_pos_tagged.csv'.")
        print("Asegurar de que el archivo existe'")
        df = None

    # 4. Frase de ejemplo
    frase_ejemplo = "El bajo precio del hotel es bajo comparado con otros."
    print("\nCOMPARACIÓN EN FRASE CON AMBIGÜEDAD")
    df_comp = comparar_etiquetas_en_frase(frase_ejemplo, tagger_nltk, nlp)
    print(df_comp.to_string(index=False))

    # 5. Ejemplo real si existe CSV
    if df is not None:
        print("\nEJEMPLO DE RESEÑA REAL (con las primeras 5 palabras)")
        primera_fila = df.iloc[0]
        texto_corto = primera_fila['reseña'][:100] + "..."
        print(f"Texto: {texto_corto}")
        print("spaCy:", primera_fila['tokens_spacy'][:5])
        print("NLTK:  ", primera_fila['tokens_nltk'][:5])