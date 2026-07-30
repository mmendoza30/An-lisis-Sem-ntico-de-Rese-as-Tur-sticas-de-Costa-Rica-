Esquema de nuestro Mongo para guardar la información sobre reseñas

{
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