import ast
import string
from datetime import datetime
import pandas as pd
from pymongo import MongoClient

mongo_url = "mongodb://localhost:27017"
dbname = "Proyecto2"
colname = "Proyecto2"


def get_colletion():
    client = MongoClient(mongo_url)
    db = client[dbname]
    return db[colname]


def cal_met_nlp(tags_spacy):
    if not tags_spacy:
        return {
            "num_palabras": 0,
            "densidad_lexica": 0.0,
            "ratio_sustantivos_verbos": 0.0,
            "densidad_adjetivos": 0.0
        }

    tokens = len(tags_spacy)
    cont_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    palabras = 0
    sus = 0
    verbos = 0
    adj = 0

    for item in tags_spacy:
        # Extraer la etiqueta POS de la estructura token, pos
        pos = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else None

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


def agregar_reseñas(col, resena, calificacion, tipolugar, nombre, fecha, pos_tags=None, metrics=None, fuente="",
                    idioma="ES", url_fuente="webscraping"):
    # Búsqueda de duplicados usando la clave 'resena' y 'lugar'
    if col.find_one({"lugar": nombre, "resena": resena}):
        return None

    res = {
        "resena": resena,
        "calificacion": calificacion,
        "tipolugar": tipolugar,
        "lugar": nombre,
        "fuente": fuente,
        "fecha": fecha,
        "fecha_recopilacion": datetime.utcnow(),
        "idioma": idioma,
        "urlfuente": url_fuente,
        "pos_tags": pos_tags or {},
        "embeddings": {},
        "metricas": metrics or {
            "num_palabras": 0,
            "densidad_lexica": 0.0,
            "ratio_sustantivos_verbos": 0.0,
            "densidad_adjetivos": 0.0
        },
    }
    resultado = col.insert_one(res)
    return resultado.inserted_id


def migrarcion_a_mongo(ruta_csv, col, res="reseña"):
    df = pd.read_csv(ruta_csv, sep=',').fillna("")
    insertados = 0

    for i, row in df.iterrows():
        resena = str(row.get(res, ''))

        # convertimos string del CSV a listas reales de Python
        try:
            tags_spacy = ast.literal_eval(str(row.get('tokens_spacy', '[]')))
        except:
            tags_spacy = []

        try:
            tags_nltk = ast.literal_eval(str(row.get('tokens_nltk', '[]')))
        except:
            tags_nltk = []

        # Se calcula métricas directo de las listas leídas
        res_metricas = cal_met_nlp(tags_spacy)

        calificacion_val = str(row.get('calificación', 'N/A'))
        tipo_lugar_val = str(row.get('tipo_lugar', 'Desconocido'))
        fuente_val = str(row.get('fuente', 'Desconocido'))
        fecha_val = str(row.get('fecha', ''))
        lugar_val = str(row.get('nombre', ''))

        _id = agregar_reseñas(
            col,
            resena=resena,
            calificacion=calificacion_val,
            tipolugar=tipo_lugar_val,
            nombre=lugar_val,
            fecha=fecha_val,
            fuente="csv",
            url_fuente=fuente_val,
            pos_tags={"nltk": tags_nltk, "spacy": tags_spacy},
            metrics=res_metricas
        )
        if _id:
            insertados += 1

    print(f"Migradas {insertados} reseñas a MongoDB.")
    return insertados


## Consultas a realizar en Mongo
def filtrar_lugar(col, lugar):
    result = list(col.find({"lugar": lugar}, {"_id": 0}))
    print(f"\n Reseñas de '{lugar}': {len(result)}")
    for c in result[:3]:
        print(f"   - {c.get('resena')} — {c.get('calificacion')} ({c.get('fecha')})")
    if len(result) > 3:
        print(f"   ... y {len(result) - 3} más")
    return result


def filtrar_fuente(col, fuente):
    result = list(col.find({"fuente": fuente}, {"_id": 0}))
    print(f"\n Reseñas de fuente '{fuente}': {len(result)}")
    for c in result[:3]:
        print(f"   - {c.get('resena')} — {c.get('calificacion')} ({c.get('fecha')})")
    if len(result) > 3:
        print(f"   ... y {len(result) - 3} más")
    return result


def busqueda_tipolugar(col, tipo_lugar):
    result = list(col.find(
        {"tipolugar": {"$regex": tipo_lugar, "$options": "i"}},
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
            "_id": "$tipolugar",
            "total_resenas": {"$sum": 1},
            "fuentes": {"$addToSet": "$fuente"}
        }},
        {"$sort": {"total_resenas": -1}}
    ]
    result = list(col.aggregate(pipeline))
    print("\n Resumen por tipo de lugar:")
    for r in result:
        print(f"   {r['_id']:15} → {r['total_resenas']:,} reseñas | fuentes: {r['fuentes']}")
    return result