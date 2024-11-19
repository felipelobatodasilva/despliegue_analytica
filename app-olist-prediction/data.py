# data.py
import pandas as pd
import joblib
from functools import lru_cache

# Cargar y procesar los datos con lru_cache
@lru_cache(maxsize=1)
def cargar_datos():
    # Cargando los datos
    ruta_archivo = 'data/base_dash_final.csv'
    df = pd.read_csv(ruta_archivo)

    # Realizar preprocesamiento necesario
    df['fecha_pedido'] = pd.to_datetime(df['fecha_pedido'])
    df['nombre_categoria_producto'] = df['nombre_categoria_producto'].fillna('Desconocido')
    df['predicted_price'] = df['predicted_price'].fillna(0)

    return df

# Cargar el modelo de regresión lineal
@lru_cache(maxsize=1)
def cargar_modelo():
    modelo_path = 'data/modelo_regresion_lineal.pkl'
    modelo_regresion_lineal = joblib.load(modelo_path)
    return modelo_regresion_lineal

# Funciones para obtener datos específicos
def get_df():
    return cargar_datos()

def get_modelo():
    return cargar_modelo()

def get_demanda_df():
    df = get_df()
    categoria_frecuencia = df['nombre_categoria_producto'].value_counts().reset_index()
    categoria_frecuencia.columns = ['nombre_categoria_producto', 'frecuencia']
    categoria_frecuencia['demanda'] = categoria_frecuencia['frecuencia'].apply(
        lambda x: 'Alta' if x > 30 else ('Media' if x > 10 else 'Baja')
    )
    demanda_df = categoria_frecuencia[['nombre_categoria_producto', 'demanda']]
    return demanda_df

def get_categorias_ordenadas():
    df = get_df()
    return sorted(df['nombre_categoria_producto'].unique())

def get_temporadas_ordenadas():
    df = get_df()
    return sorted(df['temporada'].unique())

def get_ciudades_ordenadas():
    df = get_df()
    return sorted(df['ciudad_cliente'].unique())

def get_ids_productos_ordenados():
    df = get_df()
    return sorted(df['id_producto'].unique())

def get_data_final():
    df = get_df()
    return df['fecha_pedido'].max()
