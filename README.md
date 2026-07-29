# Análisis Semántico de Reseñas Turísticas de Costa Rica
> **Aplicando Word2Vec, BETO y MongoDB**  
> **Curso:** Minería de Textos — Colegio Universitario de Cartago 
> **Profesor:** Osvaldo González Chaves  
> **Proyecto 2**

---

## 📋 Descripción del Proyecto

Este proyecto extiende la fase morfosintáctica (Proyecto 1) incorporando representaciones semánticas avanzadas del lenguaje natural. A través de **Word2Vec** (embeddings estáticos) y **BETO** (*Spanish BERT*, embeddings contextuales), se descubren relaciones de significado profundas, campos semánticos y patrones de opinión en reseñas turísticas de parques, hoteles y restaurantes de Costa Rica. 

Además, se implementa una arquitectura documental **NoSQL con MongoDB** para la gestión del corpus y un módulo de **Web Scraping** para enriquecer el conjunto de datos con nuevas experiencias de usuarios.

---

## ✨ Características Principales

* **Estructura NoSQL Enriquecida:** Almacenamiento semi-estructurado en **MongoDB** con metadatos de ubicación, polaridad, idioma, métricas léxicas, POS Tags y representaciones vectoriales.
* **Pipeline de Web Scraping:** Módulo para extracción automatizada de reseñas desde fuentes turísticas (TripAdvisor, Booking, Google Maps) cumpliendo con políticas de *rate-limiting* y *robots.txt*.
* **Representación Vectorial Estática (Word2Vec):** Entrenamiento de modelos **CBOW** y **Skip-Gram** sobre el corpus turístico para identificar clústeres temáticos, analogías vectoriales y vocabulario exclusivo.
* **Representación Vectorial Contextual (BETO):** Embeddings profundos basados en Transformer en español (`dccuchile/bert-base-spanish-wwm-cased`), permitiendo desambiguación de polisemia, búsqueda semántica y tareas de *Masked Language Modeling* (MLM).
* **Análisis Comparativo Multi-Modelo:** Evaluación cuantitativa y cualitativa entre **Bag of Words / TF-IDF**, **Word2Vec** y **BETO** mediante tareas de clasificación (Logistic Regression / KNN), clustering (K-Means / Silhouette Score) y reducciones dimensionales (**t-SNE**).
* **Dashboard Interactivo:** Panel analítico y de exploración visual desarrollado estrictamente en **Plotly Dash**.

---

## 🛠️ Tecnologías y Herramientas

| Herramienta | Versión / Modelo | Propósito / Uso |
| :--- | :--- | :--- |
| **Python** | `3.10+` | Lenguaje de desarrollo principal |
| **MongoDB** | `Community 7.x` / `Atlas` | Base de datos NoSQL para almacenamiento documental |
| **PyMongo** | `4.x` | Driver oficial de conexión Python ↔ MongoDB |
| **Gensim** | `4.x` | Entrenamiento e inferencia de modelos Word2Vec |
| **HuggingFace Transformers** | `4.x` | Carga e inferencia del modelo pre-entrenado BETO |
| **PyTorch** | `2.x` | Backend computacional para el procesamiento con BETO |
| **scikit-learn** | `1.x` | Modelado de TF-IDF, K-Means, t-SNE, regresiones |
| **spaCy** / **NLTK** | `3.x` / `3.x` | Preprocesamiento, tokenización y filtrado de stopwords |
| **BeautifulSoup4** / **Requests** / **Selenium** | `4.x` / `2.x` / `4.x` | Extracción de datos y Web Scraping |
| **Plotly Dash** | `2.x` / `5.x` | Dashboard interactivo visual |

---

## 🏗️ Estructura del Repositorio

```text
proyecto2-analisis-semantico-resenas/
├── README.md                   # Documentación principal del proyecto
├── USO_DE_IA.md               # Registro transparente del uso de herramientas de IA
├── requirements.txt           # Dependencias de Python del proyecto
├── notebooks/                  # Notebooks interactivos de experimentación
│   ├── 01_migracion_mongodb.ipynb
│   ├── 02_web_scraping.ipynb
│   ├── 03_word2vec_analisis.ipynb
│   ├── 04_beto_analisis.ipynb
│   └── 05_comparacion_final.ipynb
├── src/                        # Código fuente modular
│   ├── db_manager.py           # Conexión y consultas a MongoDB
│   ├── scraper.py              # Extracción automatizada de reseñas
│   ├── preprocessing.py        # Limpieza, tokenización y lematización
│   ├── embeddings_w2v.py       # Funciones de entrenamiento y análisis Word2Vec
│   └── embeddings_beto.py      # Funciones de embeddings y MLM con BETO
├── dashboard/                  # Aplicación del Dashboard
│   └── app.py                  # Dashboard interactivo construido en Plotly Dash
├── data/                       # Archivos de datos
│   └── raw/                    # Archivos CSV/JSON originales (Proyecto 1)
└── docs/                       # Documentación adicional
    └── esquema_mongodb.md      # Especificación del esquema de base de datos
```

---

## 🗄️ Esquema Documental en MongoDB

Cada documento recopilado se almacena bajo una estructura flexible en la colección `resenas`:

```json
{
  "_id": { "$oid": "660f1a2b3c4d5e6f7a8b9c0d" },
  "texto": "El parque es hermoso, pero el servicio en la entrada fue bastante lento.",
  "calificacion": 4,
  "tipo_lugar": "parque",
  "lugar": "Parque Nacional Manuel Antonio",
  "polaridad": "positiva",
  "idioma": "es",
  "fuente": "tripadvisor",
  "url_fuente": "https://www.tripadvisor.com/...",
  "fecha": { "$date": "2026-03-15T00:00:00Z" },
  "fecha_recopilacion": { "$date": "2026-07-29T10:00:00Z" },
  "pos_tags": {
    "nltk": [["El", "DET"], ["parque", "NOUN"]],
    "spacy": [["El", "DET"], ["parque", "NOUN"]]
  },
  "embeddings": {
    "word2vec_avg": [0.012, -0.045, 0.128, "..."],
    "beto_cls": [-0.215, 0.431, -0.089, "..."]
  },
  "metricas": {
    "num_palabras": 13,
    "densidad_lexica": 0.61,
    "ratio_sustantivos_verbos": 1.5,
    "densidad_adjetivos": 0.15
  }
}
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/proyecto2-analisis-semantico-resenas.git
cd proyecto2-analisis-semantico-resenas
```

### 2. Crear y activar entorno virtual
```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_md
```

### 4. Configurar variables de entorno (`.env`)
Crea un archivo `.env` en la raíz del proyecto con la conexión a MongoDB:
```env
MONGO_URI=mongodb+srv://<usuario>:<password>@cluster.mongodb.net/turismo_cr?retryWrites=true&w=majority
DB_NAME=turismo_cr
COLLECTION_NAME=resenas
```

---

## 📊 Ejecución del Proyecto

### Migración y Scraping
Para cargar el corpus inicial y ejecutar el raspado web:
```bash
python -m src.db_manager
python -m src.scraper
```

### Ejecución del Dashboard (Plotly Dash)
Para visualizar la exploración de embeddings, la proyecciones t-SNE y las métricas comparativas:
```bash
python dashboard/app.py
```
Abre tu navegador en `http://127.0.0.1:8050/`.

---

## 🔬 Análisis Requeridos

### 1. Word2Vec (Representaciones Estáticas)
* **Campos Semánticos:** Identificación de clusters (Naturaleza, Servicio, Precio, Limpieza, Comida, Ubicación).
* **Analogías Vectoriales:** Ej. $	ext{hotel} - 	ext{habitación} + 	ext{plato}  pprox 	ext{restaurante}$.
* **Similitud Coseno entre Categorías:** Comparación de vectores centroidales por tipo de lugar y polaridad.

### 2. BETO (Embeddings Contextuales)
* **Desambiguación de Polisemia:** Variación del vector de términos como *"rico"*, *"fresco"*, *"caro"*, *"atención"* según contexto.
* **Búsqueda Semántica:** Motor de recuperación basado en distancia coseno sobre el vector `[CLS]`.
* **Masked Language Model (MLM):** Evaluación de predicciones contextuales (ej. *"El servicio del hotel fue [MASK]"*).

### 3. Comparativa BoW vs. Word2Vec vs. BETO
* **Clasificación:** Comparación de exactitud en predicción de polaridad y tipo de lugar.
* **Clustering:** K-Means evaluado mediante *Silhouette Score*.
* **Visualización:** Mapeo bidimensional interactivo con t-SNE.

---

## 🤝 Declaración del Uso de IA

Este proyecto promueve el uso transparente y ético de herramientas de Inteligencia Artificial. Toda interacción, prompt de consulta, refactorización y modificación de código asistida se encuentra registrada en el archivo [USO_DE_IA.md](./USO_DE_IA.md).
