from string import punctuation

import nltk
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
from data.preprocessor import tokenizacion
import string
import spacy
from nltk import word_tokenize, pos_tag
nlp = spacy.load("en_core_web_sm")

mongo_url = "mongodb://localhost:27017"
dbname = "Proyecto2"
colname = "Proyecto2"

def get_colletion():
    client = MongoClient(mongo_url)
    db = client[dbname]
    return db[colname]

def agregar_reseñas(col,resena, calificacion,tipolugar,nombre,fecha,tokens_spacy, tokens_nltk,postag=None, metrics =None, fuente="",idioma="ES",url_fuente="webscraping"):
    if col.find_one({"nombre":nombre, "resena":resena}):
        return None

    res = {
        "texto" : resena,
        "calificacion" : calificacion,
        "tipo_lugar" : tipolugar,
        "lugar" : nombre,
        "fuente": fuente,
        "fecha" : fecha,
        "fecha_recopilacion" : datetime.utcnow(),
        "idioma" : idioma,
        "url_fuente" : url_fuente,
        "pos_tags" : {},
        "embeddings" : {},
        "metricas" : metrics or {
            "num_palabras" : 0,
            "densidad_lexica": 0.0,
            "ratio_sustantivos_verbos":0.0,
            "densidad_adjetivos": 0.0
        },
    }
    resultado = col.insert_one(res)
    return resultado.inserted_id


def obtencion_nltk(resena):
    if not resena or resena == "":
        return []

    tok = word_tokenize(resena)
    pos_tags = pos_tag(tok)
    punctuation = set(string.punctuation)
    taglimpios = [[res,tag] for res, tag in pos_tags if res not in punctuation]
    return taglimpios

def obtencion_spacy(resena, nlp_model):

    if not resena or resena == "":
        return []

    res = nlp_model(resena)
    tags = [[token.text, token.pos_] for token in res if not token.is_stop]
    return tags

def cal_met_nlp(tags_spacy):
    if not tags_spacy:
        return {
            "num_palabras":0,
            "densidad_lexica":0.0,
            "ratio_sustantivos_verbos": 0.0,
            "densidad_adjetivos":0.0
        }

    tokens = len(tags_spacy)
    cont_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    palabras = 0
    sus = 0
    verbos = 0
    adj = 0

    for token, pos in tags_spacy:
        if pos in cont_pos:
            palabras += 1
        if pos in {'NOUN', 'PROPN'}:
            sus += 1
        if pos == 'VERB':
            verbos += 1
        if pos == 'ADJ':
            adj += 1

    densidad = palabras / tokens if tokens > 0 else 0
    ratio = sus / verbos if verbos > 0 else sus
    densidad_adj = adj / tokens if tokens > 0 else 0

    return {
        "num_palabras": tokens,
        "densidad_lexica": round(densidad, 2),
        "ratio_sustantivos_verbos": round(ratio, 2),
        "densidad_adjetivos": round(densidad_adj, 2)
    }


def migrarcion_a_mongo(ruta_csv, col, res="text"):
    df = pd.read_csv(ruta_csv, sep=',').fillna("")
    insertados = 0

    for i, row in df.iterrows():
        resena = str(row.get(res, ''))

        tags_nltk  = obtencion_nltk(resena)
        tags_spacy = obtencion_spacy(resena, nlp)
        res_metricas   = cal_met_nlp(tags_spacy)

        resena_val  = str(row.get('reseña', ''))
        calificacion_val = str(row.get('calificación', 'N/A'))
        tipo_lugar_val  = str(row.get('tipo_lugar', 'Desconocido'))
        fuente_val = str(row.get('fuente', 'Desconocido'))
        fecha_val = str(row.get('fecha', ''))
        lugar_val = str(row.get('nombre', ''))


        _id = agregar_reseñas(
            col,
            resena     = resena_val,
            calificacion    = calificacion_val,
            tipo_lugar     = tipo_lugar_val ,
            lugar = lugar_val,
            fecha      = fecha_val,
            fuente     = "csv",
            url_fuente = fuente_val,
            pos_tags   = {"nltk": tags_nltk, "spacy": tags_spacy},
            metricas   = res_metricas
        )
        if _id:
            insertados += 1

    print(f"Migradas {insertados} reseñas a MongoDB.")
    return

## Consultas a realizar en Mongo
def filtrar_lugar(col, lugar):
    result = list(col.find({"lugar": lugar}, {"_id": 0}))
    print(f"\n Resenas de '{lugar}': {len(result)}")
    for c in result[:3]:
        print(f"   - {c.get('resena')} — {c.get('calificacion')} ({c.get('fecha')})")
    if len(result) > 3:
        print(f"   ... y {len(result) - 3} más")
    return result

def filtrar_fuente(col, fuente):
    result = list(col.find({"fuente": fuente}, {"_id": 0}))
    print(f"\n Canciones de fuente '{fuente}': {len(result)}")
    for c in result[:3]:
        print(f"   - {c.get('resena')} — {c.get('calificacion')} ({c.get('fecha')})")
    if len(result) > 3:
        print(f"   ... y {len(result) - 3} más")
    return result


def busqueda_tipolugar(col, tipo_lugar):
    result = list(col.find(
        {"tipo_lugar": {"$regex": tipo_lugar, "$options": "i"}},
        {"_id": 0}
    ))
    print(f"\n Reseñas de '{tipo_lugar}': {len(result)}")
    for c in result[:3]:
        print(f"   - {c.get('resena')} — {c.get('calificacion')} ({c.get('fecha')})")
    if len(result) > 3:
        print(f"   ... y {len(result) - 3} más")
    return result

def cantidad_resenas_tipolugar(col):
    pipeline = [
        {"$group": {
            "_id": "$tipo_lugar",
            "total_resenas": {"$sum": 1},
            "fuentes": {"$addToSet": "$fuente"}
        }},
        {"$sort": {"total_resenas": -1}}
    ]
    result = list(col.aggregate(pipeline))
    print("\n Resumen por tipo de lugar:")
    for r in result:
        print(f"   {r['_id']:15} → {r['total_resenas']:,} resenas | fuentes: {r['fuentes']}")
    return result