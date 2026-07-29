""""
Modulo de analisis para la distribución de categorias gramaticales -POS
"""

import pandas as pd

def distribucion_pos(col_etq):
    #Se recibe la columna del DF donde esta la lista de tuplas (token, etiqueta) y se retorna el conteo de frecuencias por categorias

    conteo = {}

    for lst in col_etq:
        #Se valida que existan datos validos
        if isinstance(lst, list):
            for token, etq in lst:
                #Se estandariza la etiqueta
                tag = str(etq).upper().strip()

                #Se genera un diccionario con las frecuencias
                if tag in conteo:
                    conteo[tag] += 1
                else:
                    conteo[tag] = 1

    #Se crea un DF del diccionario de una manera estructurada
    df_diccionario = pd.DataFrame(list(conteo.items()),columns=['Categoría','Frecuencia'])

    #Se calcula la distribucion relativa por categoria
    total_etq = df_diccionario["Frecuencia"].sum()
    df_diccionario["Porcentaje"] = round((df_diccionario["Frecuencia"] / total_etq) * 100, 2)

    #Se ordena el DF de mayor a menor
    df_diccionario = df_diccionario.sort_values(by=['Frecuencia'], ascending=False).reset_index(drop=True)
    return df_diccionario