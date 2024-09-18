# Importación de librerías necesarias
from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="Recomendación de peliculas",
            description="Esta es una aplicación que permite realizar consultas sobre películas personalizadas. Creada por Daiana Salcedo (9/2024)")

# Cargar los archivos .parquet
df = pd.read_parquet("api_consultas.parquet")
df2 = pd.read_parquet("movies_recomendaciones.parquet")


# Función para obtener la cantidad de filmaciones por mes
@app.get("/cantidad_filmaciones_mes/{mes}", name = "Cantidad de peliculas por mes")
async def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    meses = {
        'enero': 'January',
        'febrero': 'February',
        'marzo': 'March',
        'abril': 'April',
        'mayo': 'May',
        'junio': 'June',
        'julio': 'July',
        'agosto': 'August',
        'septiembre': 'September',
        'octubre': 'October',
        'noviembre': 'November',
        'diciembre': 'December'
    }
    if mes not in meses:
        return f"El mes {mes} no es válido"
    mes_ingles = meses[mes]
    resultado = df[df["release_date"].dt.strftime('%B') == mes_ingles]
    cantidad = len(resultado)
    return f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes}"

# Función para obtener la cantidad de filmaciones por día
@app.get("/cantidad_filmaciones_dia/{dia}", name = "Cantidad de peliculas por día")
async def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    dias = {
        'lunes': 'Monday',
        'martes': 'Tuesday',
        'miercoles': 'Wednesday',
        'miércoles': 'Wednesday',
        'jueves': 'Thursday',
        'viernes': 'Friday',
        'sabado': 'Saturday',
        'sábado': 'Saturday',
        'domingo': 'Sunday'
    }
    if dia not in dias:
        return f"El día {dia} no es válido"
    dia_ingles = dias[dia]
    resultado = df[df["release_date"].dt.strftime('%A') == dia_ingles]
    cantidad = len(resultado)
    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}"

# Función para obtener la puntuación de una película
@app.get("/score_titulo/{titulo}", name = "Puntuación de una película")

async def score_titulo(titulo: str):
    df_copy = df.copy()
    df_copy["title_lower"] = df_copy["title"].str.lower()
    titulo_lower = titulo.lower()
    resultado = df_copy[df_copy["title_lower"] == titulo_lower]
    if resultado.empty:
        return f"No se encontró la filmación {titulo}"
    else:
        fila_max_pop = resultado.loc[resultado['popularity'].idxmax()]
        titulo = fila_max_pop["title"]
        year = int(fila_max_pop["release_year"])
        score = fila_max_pop["popularity"]
        return f"La película {titulo} fue estrenada en el año {year} con una popularidad de {score}"

# Función para obtener la cantidad de votos y el valor promedio de una película
@app.get("/votos_titulo/{titulo}", name = "Cantidad de votos de una pelicula")
async def votos_titulo(titulo: str):
    df_copy = df.copy()
    df_copy["title_lower"] = df_copy["title"].str.lower()
    titulo_lower = titulo.lower()
    resultado = df_copy[df_copy["title_lower"] == titulo_lower]
    if resultado.empty:
        return f"No se encontró la filmación {titulo}"
    else:
        year = int(resultado["release_year"].iloc[0])
        voto_total = int(resultado["vote_count"].iloc[0])
        voto_promedio = resultado["vote_average"].iloc[0]
        titulo = resultado["title"].iloc[0]
        if voto_total >= 2000:
            return f"La película {titulo} fue estrenada en el año {year}. La misma cuenta con un total de {voto_total} valoraciones, con un promedio de {voto_promedio}"
        else:
            return f"No se encontraron suficientes votos para la filmación {titulo}"

# Función para obtener la información de un actor
@app.get("/get_actor/{nombre_actor}", name = "Búsqueda por el nombre/apellido de un actor")
async def get_actor(nombre_actor: str):
    df_copy = df.copy()
    df_copy["actores"].fillna("", inplace=True)
    df_copy["actores_list"] = df_copy["actores"].str.split(",")
    nombre_actor = nombre_actor.title()
    df_filtrado = df_copy[df_copy["actores_list"].apply(lambda x: nombre_actor in [nombre.title() for nombre in x])]
    cantidad_peliculas = len(df_filtrado)
    retorno_promedio = round(df_filtrado["return"].mean(), 4)
    retorno_total = round(df_filtrado["return"].sum(), 4)
    return f"El actor {nombre_actor} ha participado de {cantidad_peliculas} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_total} con un promedio de {retorno_promedio} por filmación"

# Función para obtener la información de un director
@app.get("/get_director/{nombre_director}", name = "Búsqueda por el nombre/apellido de un director")
async def get_director(nombre_director: str):
    df_copy = df.copy()
    df_copy["director"].fillna("", inplace=True)
    df_copy["director_list"] = df_copy["director"].str.split(",")
    nombre_director = nombre_director.title()
    resultado = df_copy.loc[df_copy["director_list"].apply(lambda x: nombre_director in [nombre.title() for nombre in x])]
    if resultado.empty:
        return f"No se encontró el director {nombre_director}"
    else:
        retorno_total_director = round(resultado["return"].sum(), 4)
        peliculas = []
        for index, row in resultado.iterrows():
            pelicula = {
                'titulo': row['title'],
                'anio': row['release_year'],
                'retorno_pelicula': round(row['return'], 4),
                'budget_pelicula': row['budget'],
                'revenue_pelicula': row['revenue']
            }
            peliculas.append(pelicula)
        return {'director': nombre_director,
                'retorno_total_director': retorno_total_director,
                'peliculas': peliculas}

# Crear una instancia de la clase TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

# Aplicar la transformación TF-IDF al texto contenido en la columna "overview" de 'df2'
tfidf_matriz = tfidf.fit_transform(df2['overview_clean'])

# Función para obtener las recomendaciones de películas
@app.get('/recomendacion/{titulo}', name= "Recomendación de peliculas")
async def recomendacion(titulo: str):
    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
    idx = pd.Series(indices[titulo]) if titulo in indices else None
    if idx is None:
        return "Película no encontrada"
    if df2.duplicated(['title']).any():
        primer_idx = df2[df2['title'] == titulo].index[0]
        if not idx.equals(pd.Series(primer_idx)):
            idx = pd.Series(primer_idx)
    similitud = sorted(enumerate(cosine_similarity(tfidf_matriz[idx], tfidf_matriz).flatten()), key=lambda x: x[1], reverse=True)[1:6]
    recomendaciones = df2.iloc[[i[0] for i in similitud], :]['title'].tolist()
    return {'lista recomendada': recomendaciones}