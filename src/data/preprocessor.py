"""
Modelo de preprocesamiento y tokenización de las reseñas.
Herramientas utilizadas: NLTK y spaCy.
"""

import pandas as pd
import spacy
import nltk
from pathlib import Path


class tokenizacion:

    def __init__(self):
        # nltk.download('punkt')
        # nltk.download('cess_esp')

        # Cargar corpus en español para entrenar el etiquetador de NLTK
        from nltk.corpus import cess_esp

        entrenamiento = cess_esp.tagged_sents()

        print("Entrenando el etiquetador de NLTK para español...")
        self.tagger_nltk = nltk.UnigramTagger(entrenamiento)

        print("Cargando modelo de spaCy...")
        self.nlp = spacy.load("es_core_news_md")

    # -------------------------------------------------------
    # Tokenización con spaCy
    # -------------------------------------------------------
    def proceso_tokenizacion_spacy(self, texto):

        if pd.isna(texto) or str(texto).strip() == "":
            return []

        doc = self.nlp(str(texto))

        lista = []

        for token in doc:
            lista.append((token.text, token.pos_))

        return lista

    # -------------------------------------------------------
    # Tokenización con NLTK
    # -------------------------------------------------------
    def proceso_tokenizacion_nltk(self, texto):

        if pd.isna(texto) or str(texto).strip() == "":
            return []

        tokens = nltk.word_tokenize(str(texto), language="spanish")

        return self.tagger_nltk.tag(tokens)

    # -------------------------------------------------------
    # Aplicar tokenización al DataFrame
    # -------------------------------------------------------
    def proceso_tokenizacion(self, df, column="reseña"):

        df_rst = df.copy()

        print("[PREPROCESSOR] Aplicando tokenización con spaCy y NLTK...")

        df_rst["tokens_spacy"] = df_rst[column].apply(
            self.proceso_tokenizacion_spacy
        )

        df_rst["tokens_nltk"] = df_rst[column].apply(
            self.proceso_tokenizacion_nltk
        )

        # Ruta de salida
        ruta_salida = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "processed"
            / "reseñas_pos_tagged_P2.csv"
        )

        df_rst.to_csv(ruta_salida, index=False)

        print("Proceso de tokenización finalizado.")
        print(f"Archivo guardado en:\n{ruta_salida}")

        return df_rst


if __name__ == "__main__":

    # Archivo limpio generado por cleaner.py
    ruta_clean = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "processed"
        / "reseñas_clean_P2.csv"
    )

    if ruta_clean.exists():

        print(f"Leyendo archivo:\n{ruta_clean}\n")

        df = pd.read_csv(ruta_clean)

        preprocesador = tokenizacion()

        df_procesado = preprocesador.proceso_tokenizacion(df)

        print("\nPrimer ejemplo de spaCy:")
        print(df_procesado["tokens_spacy"].iloc[0])

        print("\nPrimer ejemplo de NLTK:")
        print(df_procesado["tokens_nltk"].iloc[0])

        print(f"\nTotal de reseñas procesadas: {len(df_procesado)}")

    else:

        print(f"No se encontró el archivo:\n{ruta_clean}")
        print("Ejecute primero cleaner.py.")