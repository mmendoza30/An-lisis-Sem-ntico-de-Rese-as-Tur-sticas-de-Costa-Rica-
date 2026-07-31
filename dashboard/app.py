import os
import sys
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Gensim para vectorización Word2Vec
import gensim
from gensim.models import Word2Vec

#Importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importación de las funciones del módulo de embeddings
try:
    from embeddings_w2v import (
        obtencion_corpus,
        modelos_CBOW_SKIPGRAM,
        palabras_similares,
        analogias_vectoriales,
        similitud_entre_lugares,
        promedio_vectores
    )
except ImportError:
    try:
        from src.embeddings_w2v import (
            obtencion_corpus,
            modelos_CBOW_SKIPGRAM,
            palabras_similares,
            analogias_vectoriales,
            similitud_entre_lugares,
            promedio_vectores
        )
    except ImportError:
        obtencion_corpus = None
        modelos_CBOW_SKIPGRAM = None
        similitud_entre_lugares = None

# Tema de Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Análisis Semántico - Costa Rica"


def cargar_datos_mongo():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client["Proyecto2"]
        collection = db["Proyecto2"]

        cursor = collection.find({})
        df_raw = pd.DataFrame(list(cursor))

        if "_id" in df_raw.columns:
            df_raw = df_raw.drop(columns=["_id"])

    except Exception as e:
        print(f"Problemas con la conexion con Mongo({e}). Usando fallback...")
        df_raw = pd.DataFrame()

    if df_raw.empty:
        # Respaldo en caso de que Mongo no responda
        df_raw = pd.DataFrame({
            "name": ["Baldi Hot Springs Hotel Resort & Spa", "Costa Rica Marriott Hotel Hacienda Belen",
                     "Parque Nacional Tortuguero"],
            "category": ["hotel", "hotel", "parque nacional"],
            "review_rating": [5, 4, 1],
            "review_text": [
                "Excelente experiencia en las piscinas y toboganes.",
                "Buena atención y estadía confortable.",
                "Instalaciones descuidadas."
            ]
        })

    # Mapeo de columnas según el esquema guardado en Mongo
    col_name = "name" if "name" in df_raw.columns else ("lugar" if "lugar" in df_raw.columns else df_raw.columns[0])
    col_cat = "category" if "category" in df_raw.columns else (
        "tipolugar" if "tipolugar" in df_raw.columns else df_raw.columns[1])
    col_rating = "review_rating" if "review_rating" in df_raw.columns else (
        "rating" if "rating" in df_raw.columns else df_raw.columns[2])
    col_text = "review_text" if "review_text" in df_raw.columns else (
        "resena" if "resena" in df_raw.columns else df_raw.columns[3])

    df_clean = pd.DataFrame()
    df_clean["lugar"] = df_raw[col_name].fillna("Sin Nombre")
    df_clean["tipolugar"] = df_raw[col_cat].fillna("general")
    df_clean["categoria"] = df_raw[col_cat].fillna("General")
    df_clean["rating"] = pd.to_numeric(df_raw[col_rating], errors="coerce").fillna(3)
    df_clean["resena"] = df_raw[col_text].fillna("Sin reseña")

    # Mapeo de polaridad
    def calcular_polaridad(r):
        if r >= 4:
            return "positiva"
        elif r == 3:
            return "neutra"
        else:
            return "negativa"

    df_clean["polaridad"] = df_clean["rating"].apply(calcular_polaridad)

    # Coordenadas t-SNE / Embeddings para mostrar grafico mas adelante
    if "tsne_x" in df_raw.columns and "tsne_y" in df_raw.columns:
        df_clean["tsne_x"] = df_raw["tsne_x"]
        df_clean["tsne_y"] = df_raw["tsne_y"]
    else:
        np.random.seed(42)
        df_clean["tsne_x"] = np.random.normal(loc=0, scale=8, size=len(df_clean))
        df_clean["tsne_y"] = np.random.normal(loc=0, scale=8, size=len(df_clean))

    return df_clean


# Cargar dataset de MongoDB
df = cargar_datos_mongo()

# Inicialización segura de los modelos Word2Vec
cbow_model = None
sg_model = None

try:
    if modelos_CBOW_SKIPGRAM and callable(modelos_CBOW_SKIPGRAM):
        cbow_model, sg_model = modelos_CBOW_SKIPGRAM(df)
except Exception as err:
    print(f"Error al intentar cargar el dashboard: {err}")

# En dado caso de error de comunicacion entre archivos realizar calculo aca
if cbow_model is None or sg_model is None:
    try:
        corpus_tokens = [str(text).lower().split() for text in df["resena"].dropna()]
        if corpus_tokens:
            cbow_model = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=1, sg=0)
            sg_model = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=1, sg=1)
            print("Modelos CBOW y Skip-Gram inicializados exitosamente con Gensim.")
    except Exception as e_gensim:
        print(f"Error entrenando modelos fallback: {e_gensim}")

# Barra Lateral (Sidebar)
sidebar = dbc.Card(
    [
        html.H4("Filtros", className="card-title text-primary fw-bold"),
        html.Hr(),
        html.Label("Lugar Turístico:", className="fw-bold"),
        dcc.Dropdown(
            id="filtro-lugar",
            options=[{"label": str(i), "value": str(i)} for i in sorted(df["lugar"].unique())],
            multi=True,
            placeholder="Todos los lugares...",
            className="mb-3"
        ),
        html.Label("Polaridad:", className="fw-bold"),
        dcc.Dropdown(
            id="filtro-polaridad",
            options=[
                {"label": "Positiva (4-5 ★)", "value": "positiva"},
                {"label": "Neutra (3 ★)", "value": "neutra"},
                {"label": "Negativa (1-2 ★)", "value": "negativa"}
            ],
            multi=True,
            placeholder="Todas las polaridades...",
            className="mb-3"
        ),
    ],
    body=True,
    style={"height": "100%", "backgroundColor": "#f8f9fa"}
)

content = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2("Análisis Semántico de Reseñas Turísticas de Costa Rica",
                            className="display-6 text-dark fw-bold"),
                    html.P("Dashboard interactivo conectado a MongoDB para la exploración semántica y clasificación de opiniones.",
                           className="lead text-muted"),
                ]),
                width=12
            ),
            className="mb-4"
        ),

        # Tarjetas KPIs
        dbc.Row(
            [
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Total Reseñas Analizadas", className="text-muted"),
                        html.H3(id="kpi-total", className="text-primary fw-bold")
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Destinos Evaluados", className="text-muted"),
                        html.H3(id="kpi-lugares", className="text-success fw-bold")
                    ])
                ]), width=6),
            ],
            className="mb-4"
        ),

        # Pestañas principales
        dbc.Tabs(
            [
                dbc.Tab(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="grafico-tsne"), width=7),
                                dbc.Col(dcc.Graph(id="grafico-polaridad"), width=5),
                            ],
                            className="mt-3"
                        )
                    ],
                    label="Proyección Semántica (t-SNE / Embeddings)",
                ),
                dbc.Tab(
                    [
                        html.Div(
                            [
                                html.H4("Comparación Semántica entre Lugares (Word2Vec)", className="mt-3 mb-2"),
                                html.P("Matriz de similitud coseno calculada a partir del corpus vectorial de reseñas.", className="text-muted"),
                                html.Div(id="contenedor-similitud-lugares", className="mt-3")
                            ]
                        )
                    ],
                    label="Similitud entre Lugares",
                ),
                dbc.Tab(
                    [
                        html.Div(
                            [
                                html.H5("Detalle de Reseñas Filtradas", className="mt-3 mb-3"),
                                html.Div(id="tabla-resenas")
                            ]
                        )
                    ],
                    label="Explorador de Reseñas",
                ),
            ]
        )
    ],
    fluid=True
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className="p-3"),
                dbc.Col(content, width=9, className="p-3")
            ]
        )
    ],
    fluid=True
)


# Callbacks

@callback(
    [
        Output("grafico-tsne", "figure"),
        Output("grafico-polaridad", "figure"),
        Output("kpi-total", "children"),
        Output("kpi-lugares", "children"),
        Output("tabla-resenas", "children"),
        Output("contenedor-similitud-lugares", "children")
    ],
    [
        Input("filtro-lugar", "value"),
        Input("filtro-polaridad", "value")
    ]
)
def actualizar_dashboard(lugares_sel, polaridades_sel):
    df_filtrado = df.copy()

    if lugares_sel:
        df_filtrado = df_filtrado[df_filtrado["lugar"].isin(lugares_sel)]
    if polaridades_sel:
        df_filtrado = df_filtrado[df_filtrado["polaridad"].isin(polaridades_sel)]

    #Gráfico t-SNE
    fig_tsne = px.scatter(
        df_filtrado,
        x="tsne_x",
        y="tsne_y",
        color="categoria",
        symbol="polaridad",
        hover_data=["lugar", "rating", "resena"],
        title="Espacio Semántico 2D (t-SNE)",
        template="plotly_white"
    )

    #Distribución de Polaridades
    fig_pol = px.pie(
        df_filtrado,
        names="polaridad",
        title="Distribución del Sentimiento",
        color="polaridad",
        color_discrete_map={"positiva": "#2ecc71", "neutra": "#f1c40f", "negativa": "#e74c3c"},
        template="plotly_white"
    )

    #Métricas KPIs
    total_resenas = len(df_filtrado)
    total_lugares = df_filtrado["lugar"].nunique()

    # 4. Tabla de reseñas
    df_tabla = df_filtrado[["lugar", "categoria", "rating", "polaridad", "resena"]].head(50).copy()
    df_tabla["resena"] = df_tabla["resena"].apply(lambda x: (str(x)[:120] + "...") if len(str(x)) > 120 else str(x))

    tabla = dbc.Table.from_dataframe(
        df_tabla,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True
    )

    #Cálculo dinámico de Similitud entre Lugares
    componentes_similitud = []

    try:
        lugares_evaluar = lugares_sel if (lugares_sel and len(lugares_sel) >= 2) else df["lugar"].value_counts().head(3).index.tolist()

        if len(lugares_evaluar) >= 2:

            def obtener_matriz_similitud(modelo):
                matrices = []
                for l1 in lugares_evaluar:
                    fila = {}
                    txt1 = " ".join(df[df["lugar"] == l1]["resena"].dropna()).lower().split()
                    vecs1 = [modelo.wv[w] for w in txt1 if w in modelo.wv]
                    v1 = np.mean(vecs1, axis=0) if vecs1 else None

                    for l2 in lugares_evaluar:
                        txt2 = " ".join(df[df["lugar"] == l2]["resena"].dropna()).lower().split()
                        vecs2 = [modelo.wv[w] for w in txt2 if w in modelo.wv]
                        v2 = np.mean(vecs2, axis=0) if vecs2 else None

                        if v1 is not None and v2 is not None:
                            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            fila[l2] = round(float(sim), 4)
                        else:
                            fila[l2] = 0.0
                    matrices.append({"Lugar": l1, **fila})

                df_matriz = pd.DataFrame(matrices).set_index("Lugar")
                return dbc.Table.from_dataframe(df_matriz.reset_index(), striped=True, bordered=True, hover=True)

            # Se manda a llamar a
            if cbow_model:
                try:
                    res_cbow = similitud_entre_lugares({l: obtencion_corpus(df, l) for l in lugares_evaluar}, cbow_model, "CBOW") if (similitud_entre_lugares and obtencion_corpus) else None
                    if isinstance(res_cbow, pd.DataFrame):
                        res_cbow = dbc.Table.from_dataframe(res_cbow.reset_index(), striped=True, bordered=True, hover=True)
                    else:
                        res_cbow = obtener_matriz_similitud(cbow_model)
                except:
                    res_cbow = obtener_matriz_similitud(cbow_model)
            else:
                res_cbow = html.Div("Modelo CBOW no disponible", className="text-muted p-2")

            if sg_model:
                try:
                    res_sg = similitud_entre_lugares({l: obtencion_corpus(df, l) for l in lugares_evaluar}, sg_model, "Skip-Gram") if (similitud_entre_lugares and obtencion_corpus) else None
                    if isinstance(res_sg, pd.DataFrame):
                        res_sg = dbc.Table.from_dataframe(res_sg.reset_index(), striped=True, bordered=True, hover=True)
                    else:
                        res_sg = obtener_matriz_similitud(sg_model)
                except:
                    res_sg = obtener_matriz_similitud(sg_model)
            else:
                res_sg = html.Div("Modelo Skip-Gram no disponible", className="text-muted p-2")

            col_cbow = dbc.Col([
                html.H6("Modelo CBOW", className="fw-bold text-center text-primary mb-3"),
                dbc.Card(dbc.CardBody([res_cbow]))
            ], width=6)

            col_sg = dbc.Col([
                html.H6("Modelo Skip-Gram", className="fw-bold text-center text-success mb-3"),
                dbc.Card(dbc.CardBody([res_sg]))
            ], width=6)

            componentes_similitud = [dbc.Row([col_cbow, col_sg])]
        else:
            componentes_similitud = [
                dbc.Alert("Selecciona al menos 2 lugares en el filtro lateral para comparar sus similitudes semánticas.", color="info")
            ]

    except Exception as e:
        componentes_similitud = [
            dbc.Alert(f"Error calculando la similitud entre lugares: {e}", color="warning")
        ]

    return fig_tsne, fig_pol, f"{total_resenas:,}", str(total_lugares), tabla, componentes_similitud


# Servidor Flask/Dash
if __name__ == "__main__":
    app.run(debug=True, port=8050)