import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


# Limpieza basica de texto
def proceso_limpieza_word2vec(resena):
    if not resena:
        return []
    doc = nlp(resena.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and len(token.text) > 2
    ]
    return tokens


# Funcion para entrenar los modelos
def modelos_CBOW_SKIPGRAM(corpus, size, window, sg, min_count, workers):
    # Corregido: Word2Vec en Gensim usa mayusculas
    modelo = word2vec.Word2Vec(
        sentences=corpus,
        vector_size=size,
        min_count=min_count,
        window=window,
        sg=sg,
        workers=workers,
    )
    return modelo


# Filtrado de texto por lugar o dataset completo
def obtencion_corpus(data, lugar=None):
    corpus = []

    if lugar is None:
        datos_a_procesar = data["resena"]
        print("Obteniendo corpus completo...")
    else:
        datos_a_procesar = data[data["lugar"] == lugar]["resena"]
        print(f"Obteniendo corpus para: {lugar}...")

    for res in datos_a_procesar:
        corpus_limpio = proceso_limpieza_word2vec(res)
        corpus.append(corpus_limpio)

    return corpus


# Busca palabras mas cercanas en ambos modelos
def palabras_similares(modelo_cbow, modelo_sg, lugar, palabra, top=5):
    print(f"--- Campo semantico para: {lugar.upper()} ---")
    print(f"Palabra clave: '{palabra}'\n")

    print("CBOW:")
    try:
        res_cbow = modelo_cbow.wv.most_similar(palabra, topn=top)
        for p, sim in res_cbow:
            print(f" - {p}: {sim:.4f}")
    except KeyError:
        print(f"La palabra '{palabra}' no esta en el vocabulario CBOW.")

    print("\nSKIP-GRAM:")
    try:
        vecinos_s = modelo_sg.wv.most_similar(palabra, topn=top)
        for p, sim in vecinos_s:
            print(f" - {p}: {sim:.4f}")
    except KeyError:
        print(f"La palabra '{palabra}' no esta en el vocabulario Skip-Gram.")
    print("-" * 40)


# Prueba de analogias (vector_b - vector_a + vector_c)
def analogias_vectoriales(
    modelo_cbow, modelo_sg, origen_a, destino_b, base_c, top_n=3
):
    print(
        f"Analogia: {origen_a} es a {destino_b} como {base_c} es a..."
    )

    for nombre, model in [("CBOW", modelo_cbow), ("SKIP-GRAM", modelo_sg)]:
        print(f"\n[{nombre}]")
        try:
            resultados = model.wv.most_similar(
                positive=[destino_b, base_c], negative=[origen_a], topn=top_n
            )
            for i, (palabra, sim) in enumerate(resultados):
                marca = "*" if i == 0 else " "
                print(f" {marca} {palabra} ({sim:.4f})")
        except KeyError as e:
            print(f" El termino {e} no aparece en el modelo.")
    print("=" * 40)


# Similitud entre lugares mediante vectores promedio y heatmap
def similitud_entre_lugares(lugares, modelo, nombre_modelo="Word2Vec"):
    nombres = list(lugares.keys())
    vectores = []

    # Sacar vector promedio por cada lugar
    for lugar, corpus in lugares.items():
        vecs = []
        for doc in corpus:
            for word in doc:
                if word in modelo.wv:
                    vecs.append(modelo.wv[word])

        if len(vecs) > 0:
            vectores.append(np.mean(vecs, axis=0))
        else:
            print(f"Sin datos suficientes para: '{lugar}'")
            vectores.append(np.zeros(modelo.vector_size))

    # Matriz de coseno
    matriz_sim = cosine_similarity(vectores)

    print(f"\nSimilitud entre lugares ({nombre_modelo.upper()}):")
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            sim = matriz_sim[i][j]
            if sim > 0.8:
                interp = "Muy alta"
            elif sim > 0.6:
                interp = "Alta"
            elif sim > 0.4:
                interp = "Media"
            else:
                interp = "Baja"

            print(f"{nombres[i]} vs {nombres[j]} -> {sim:.4f} ({interp})")

    # Grafica Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matriz_sim,
        annot=True,
        fmt=".3f",
        xticklabels=nombres,
        yticklabels=nombres,
        cmap="YlOrRd",
    )
    plt.title(f"Similitud entre Lugares - {nombre_modelo}")
    plt.tight_layout()
    plt.show()

def promedio_vectores(resena, modelo):
    tokens = proceso_limpieza_word2vec(resena)

    #Se extraen los vectores
    vectores = [modelo.wv[p] for p in tokens if p in modelo.wv]

    #Si existen valores se realiza el promedio
    if vectores:
        return np.mean(vectores, axis=0).tolist()
    return []