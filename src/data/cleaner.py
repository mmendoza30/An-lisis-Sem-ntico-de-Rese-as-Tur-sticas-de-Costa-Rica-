"""
Módulo de limpieza de reseñas turísticas en español.
Implementa una clase ReviewCleaner que encapsula todas las transformaciones
necesarias para dejar el corpus listo para tokenización y POS tagging.
"""

import re  # Limpia texto usando patrones como URLs, caracteres especiales y espacios repetidos.
import pandas as pd  # Se utiliza para cargar, modificar y analizar el dataset de reseñas mediante DataFrames.
import numpy as np  # Facilita cálculos numéricos y el manejo de valores faltantes o especiales como NaN.
from pathlib import Path  # Ayuda a leer y guardar archivos de forma segura y compatible con distintos sistemas operativos.
from langdetect import detect, LangDetectException  # Detecta automáticamente el idioma de una reseña y permite manejar errores si no puede identificarse.
from typing import List, Optional, Dict  # Permite especificar tipos de datos esperados en funciones y métodos para mejorar la legibilidad y el mantenimiento del código.

class ReviewCleaner:
    """
    Clase para limpieza de reseñas turísticas.
    Encapsula selección de columnas, filtrado de idioma, normalización de texto,
    manejo de fechas, traducción de categorías y guardado del dataset limpio.
    """

    # Constantes de clase (mapeos y configuraciones fijas)

    COLUMNAS_DESEADAS = [
        'review_text',
        'review_rating',
        'category',
        'reviews_link',
        'name',
        'review_datetime_utc',
        'review_questions_Travel group'
    ]

    MAPEO_NOMBRES = {
        'review_text': 'reseña',
        'review_rating': 'calificación',
        'category': 'tipo_lugar',
        'reviews_link': 'fuente',
        'name': 'nombre',
        'review_datetime_utc': 'fecha',
        'review_questions_Travel group': 'grupo_de_viaje'
    }

    MESES_ES = {
        1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
        5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
        9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
    }

    MAPEO_GRUPO_VIAJE = {
        'family': 'familia',
        'couple': 'pareja',
        'friends': 'amigos'
    }

    MAPEO_TIPO_LUGAR = {
        'retaurante': 'restaurante',
        'restaurant': 'restaurante'
    }


    # Constructor y utilidades internas

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el limpiador con una copia del DataFrame original.
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame crudo recién cargado desde el CSV de entrada.
        """
        self.df = df.copy()
        self.cleaning_log: List[str] = []
        self._log("Inicializado ReviewCleaner")

    def _log(self, mensaje: str) -> None:
        """Registra un mensaje en el log interno y lo imprime en consola."""
        print(f"[CLEANER] {mensaje}")
        self.cleaning_log.append(mensaje)

    # Método para reparar fechas ambiguas

    @staticmethod
    def _reparar_fecha(fecha_str: str) -> pd.Timestamp:
        """
        Interpreta una fecha con formato ambiguo (día/mes/año o mes/día/año).
        Si el segundo componente es > 12, invierte día y mes.
        """
        if not isinstance(fecha_str, str):
            return pd.NaT
        partes = fecha_str.split('/')
        if len(partes) != 3:
            return pd.NaT
        try:
            v1, v2, v3 = int(partes[0]), int(partes[1]), int(partes[2])
        except ValueError:
            return pd.NaT

        dia, mes, anio = v1, v2, v3
        if mes > 12:
            dia, mes = v2, v1  # estaba en formato mes/día/año

        if not (1 <= mes <= 12 and 1 <= dia <= 31 and 2000 <= anio <= 2100):
            return pd.NaT
        return pd.Timestamp(year=anio, month=mes, day=dia)

    # 1. Selección de columnas relevantes

    def seleccionar_columnas(self) -> None:
        """Conserva únicamente las columnas definidas en COLUMNAS_DESEADAS."""
        columnas_presentes = [c for c in self.COLUMNAS_DESEADAS if c in self.df.columns]
        columnas_faltantes = [c for c in self.COLUMNAS_DESEADAS if c not in self.df.columns]
        self.df = self.df[columnas_presentes].copy()
        self._log(f"Columnas conservadas: {self.df.columns.tolist()}")
        if columnas_faltantes:
            self._log(f"[AVISO] Columnas no encontradas en el dataset: {columnas_faltantes}")


    # 2. Eliminación de filas totalmente vacías

    def eliminar_filas_vacias(self) -> None:
        """Elimina filas donde todas las celdas son NaN o cadenas vacías."""
        antes = len(self.df)
        vacias = (
            self.df.isnull().all(axis=1) |
            self.df.astype(str).apply(lambda x: x.str.strip() == '', axis=1).all(axis=1)
        )
        self.df = self.df[~vacias].copy()
        self._log(f"Filas completamente vacías eliminadas: {antes - len(self.df)}")


    # 3. Renombrar columnas a español

    def renombrar_columnas(self) -> None:
        """Renombra las columnas según MAPEO_NOMBRES."""
        self.df.rename(
            columns={k: v for k, v in self.MAPEO_NOMBRES.items() if k in self.df.columns},
            inplace=True
        )
        self._log(f"Columnas renombradas: {self.df.columns.tolist()}")


    # 4. Convertir texto a minúsculas

    def convertir_minusculas(self) -> None:
        """Convierte a minúsculas todas las celdas que sean cadenas de texto."""
        self.df = self.df.map(lambda x: x.lower() if isinstance(x, str) else x)
        self._log("Texto convertido a minúsculas")


    # 5. Limpiar fechas y crear columna 'mes'

    def limpiar_fechas(self) -> None:
        """
        - Quita la hora de la columna 'fecha'.
        - Interpreta fechas ambiguas (d/m/a o m/d/a) y las normaliza.
        - Formatea a dd/mm/yyyy.
        - Crea la columna 'mes' con el nombre del mes en español.
        """
        self.df['fecha'] = self.df['fecha'].str.split(' ').str[0]
        fechas_dt = self.df['fecha'].apply(self._reparar_fecha)
        self.df['fecha'] = fechas_dt.dt.strftime('%d/%m/%Y')
        self.df['mes'] = fechas_dt.dt.month.map(self.MESES_ES)
        nulos = fechas_dt.isna().sum()
        self._log(f"Fechas procesadas. Nulas (no interpretables): {nulos}")

    # 6. Eliminar reseñas vacías

    def eliminar_resenas_vacias(self) -> None:
        """Elimina filas donde la reseña es NaN, vacía o solo espacios."""
        antes = len(self.df)
        condicion = self.df['reseña'].isna() | (self.df['reseña'].str.strip() == '')
        self.df = self.df[~condicion].copy()
        self._log(f"Reseñas vacías eliminadas: {antes - len(self.df)}")


    # 7. Filtrar solo reseñas en español

    def filtrar_espanol(self) -> None:
        """Detecta el idioma de cada reseña y conserva solo las que están en español."""
        def _es_espanol(texto: str) -> bool:
            try:
                return detect(texto) == 'es'
            except LangDetectException:
                return False

        antes = len(self.df)
        mascara = self.df['reseña'].apply(_es_espanol)
        self.df = self.df[mascara].copy()
        self._log(f"Reseñas no español eliminadas: {antes - len(self.df)}")


    # 8. Limpiar texto de reseñas (emojis, símbolos, puntuación)

    def limpiar_texto_resenas(self) -> None:
        """
        Deja las reseñas con solo letras (incluyendo tildes y ñ), números y espacios.
        Elimina emojis, símbolos y signos de puntuación.
        """
        def _limpiar(texto: str) -> str:
            if not isinstance(texto, str):
                return ""
            # Quitar emojis y símbolos no latinos (conserva puntuación básica)
            texto = re.sub(r'[^\w\s.,!?;:áéíóúüñÁÉÍÓÚÜÑ]', '', texto, flags=re.UNICODE)
            # Quitar signos de puntuación
            texto = re.sub(r'[.,!?;:]', '', texto)
            # Dejar solo letras (con tildes, ñ) y números
            texto = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s]', '', texto, flags=re.UNICODE)
            # Normalizar espacios
            texto = re.sub(r'\s+', ' ', texto).strip()
            return texto

        self.df['reseña'] = self.df['reseña'].apply(_limpiar)

        # Eliminar reseñas que quedaron vacías después de la limpieza
        antes = len(self.df)
        self.df = self.df[self.df['reseña'] != ''].copy()
        self._log(f"Texto de reseñas limpiado. Vacías post-limpieza: {antes - len(self.df)}")


    # 9. Convertir calificación a entero

    def convertir_calificacion_entero(self) -> None:
        """Convierte la columna 'calificación' a tipo entero."""
        self.df['calificación'] = pd.to_numeric(
            self.df['calificación'], errors='coerce'
        ).astype('Int64')
        self._log("Calificación convertida a entero")

    # 10. Traducir categorías de grupo de viaje

    def traducir_grupo_viaje(self) -> None:
        """Traduce los valores de 'grupo_de_viaje' al español."""
        antes = self.df['grupo_de_viaje'].value_counts(dropna=False).to_dict()
        self.df['grupo_de_viaje'] = self.df['grupo_de_viaje'].replace(self.MAPEO_GRUPO_VIAJE)
        self._log(f"Grupo de viaje traducido. Distribución anterior: {antes}")

    def limpiar_tipo_lugar(self) -> None:
        """Se corriguen los errores de escritura"""
        if 'tipo_lugar' in self.df.columns:
            self.df['tipo_lugar'] = self.df['tipo_lugar'].replace(self.MAPEO_TIPO_LUGAR)
            self._log("Columna de Tipo Lugar corregida")

    # 11. Verificación final y guardado

    def verificar_estado(self) -> None:
        """Muestra un resumen del estado actual del DataFrame."""
        self._log(f"Dimensiones finales: {self.df.shape}")
        self._log(f"Columnas: {self.df.columns.tolist()}")
        self._log(f"Nulos por columna:\n{self.df.isnull().sum().to_string()}")

    def save_cleaned_data(self, ruta_salida: str) -> None:
        """
        Guarda el DataFrame limpio en un archivo CSV.

        Parámetros
        ----------
        ruta_salida : str
            Ruta completa donde guardar el archivo.
        """
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(ruta_salida, index=False)
        self._log(f"Dataset limpio guardado en: {ruta_salida}")

    # Orquestador: ejecuta todas las etapas en orden

    def ejecutar_limpieza_completa(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de limpieza en el orden adecuado.
        Retorna el DataFrame limpio.
        """
        self.seleccionar_columnas()
        self.eliminar_filas_vacias()
        self.renombrar_columnas()
        self.convertir_minusculas()
        self.limpiar_fechas()
        self.eliminar_resenas_vacias()
        self.filtrar_espanol()
        self.limpiar_texto_resenas()
        self.convertir_calificacion_entero()
        self.traducir_grupo_viaje()
        self.limpiar_tipo_lugar()
        self.verificar_estado()
        return self.df

    # Métodos de acceso

    def get_cleaning_log(self) -> List[str]:
        """Devuelve el registro completo de limpieza."""
        return self.cleaning_log

    def get_clean_dataframe(self) -> pd.DataFrame:
        """Devuelve el DataFrame limpio actual."""
        return self.df


# Bloque de ejecución directa

if __name__ == "__main__":
    # Localizar la raíz del proyecto (busca la carpeta 'data')
    proyecto = Path(__file__).resolve().parent.parent.parent  # src/data -> raíz
    if not (proyecto / 'data').exists():
        proyecto = Path.cwd()
        while not (proyecto / 'data').exists() and proyecto.parent != proyecto:
            proyecto = proyecto.parent

    ruta_entrada = proyecto / 'data' / 'raw' / 'final' / 'resenas_google_maps.csv'
    ruta_salida = proyecto / 'data' / 'processed' / 'resenas_clean.csv'

    print(f"Cargando datos desde: {ruta_entrada}")
    df_crudo = pd.read_csv(ruta_entrada)

    cleaner = ReviewCleaner(df_crudo)
    cleaner.ejecutar_limpieza_completa()
    cleaner.save_cleaned_data(str(ruta_salida))

    print("\n=== REGISTRO DE LIMPIEZA ===")
    for entrada in cleaner.get_cleaning_log():
        print(entrada)