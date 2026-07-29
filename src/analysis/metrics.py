"""
Modulo de calculo de metricas derivadas (ratio N/V, densidad léxica)
"""
import numpy as np
import pandas as pd

def calculo_radio(tuplas):
    #Se realiza el calculpo de la relacion entre sustantivos y verbos
    if not tuplas or len(tuplas) == 0:
        return np.nan
    verbos = 0
    sustantivos = 0

    for token, etq in tuplas:
        etqm = str(etq).lower().strip()
        #Procesos de identificacion de sustantivos dependientdo del tagger
        if "noun" in etqm or "propn" in etqm or etqm in ["nc", "np", "n"]:
            sustantivos += 1
        elif "verb" in etqm or "aux" in etqm or etqm in ["v", "vm", "vs"]:
            verbos += 1

    if verbos == 0:
        return float(sustantivos)

    return round(float(sustantivos) / float(verbos), 2)

def calculo_densidad_lex(tuplas):
    #Se realiza el calculo de la proporcion de las palabras con una carga semantica al total
    if not tuplas or len(tuplas) == 0:
        return np.nan
    pal = 0
    total_pal = len(tuplas)

    etq_con = ["NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "NC", "NP", "V", "AQ", "RG", "RN"]

    for token, etq in tuplas:
        #Se valida que la etiqueta limpia califica como una palabra de contenido
        contenido = False
        etq_may = str(etq).upper()

        for et in etq_con:
            if et in etq_may:
                contenido = True
                break
        if contenido:
            pal += 1
    #Se realiza el calculo de la densidad lexica
    return round(float(pal) / float(total_pal), 2)

