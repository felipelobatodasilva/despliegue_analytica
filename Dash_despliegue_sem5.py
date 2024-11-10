#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:
from dash.dependencies import Input, Output


import pandas as pd

# Cargar datos
ruta_archivo = '/home/ubuntu/mi_proyecto_dash/files_csv/base_dash.csv'

df = pd.read_csv(ruta_archivo)

# Mostrar primeras filas
df.head()


# In[3]:




# In[3]:





# In[2]:

import dash
from  dash import Dash,dcc, html
import dash.dash_table as dash_table
import pandas as pd


# Limpiar los datos (si es necesario)
df['nombre_categoria_producto'] = df['nombre_categoria_producto'].fillna('Desconocido')
df['predicted_price'] = df['predicted_price'].fillna(0)

# Clasificación de demanda según la frecuencia de la categoría
categoria_frecuencia = df['nombre_categoria_producto'].value_counts().reset_index()
categoria_frecuencia.columns = ['nombre_categoria_producto', 'frecuencia']
categoria_frecuencia['demanda'] = categoria_frecuencia['frecuencia'].apply(
    lambda x: 'Alta' if x > 30 else ('Media' if x > 10 else 'Baja')
)

# Crear la tabla de demanda
demanda_df = categoria_frecuencia[['nombre_categoria_producto', 'demanda']]

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div([

    # Título principal
    html.H1("Análisis de Categorías y Predicciones de Precio Promedio", style={'text-align': 'center', 'color': '#2f4f4f'}),

    # Sección 1: Tabla de Demanda
    html.Div([
        html.H3("Demanda de Categorías por Frecuencia", style={'text-align': 'center', 'color': '#2f4f4f'}),
        dash_table.DataTable(
            id='tabla-demanda',
            columns=[{"name": i, "id": i} for i in demanda_df.columns],
            data=demanda_df.to_dict('records'),
            style_table={'height': '350px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
        ),
    ], style={'padding': '20px'}),  # Sección 1

    # Sección 2: Gráfico de Demanda por Categoría
    html.Div([
        html.Label("Selecciona una Temporada:", style={'font-weight': 'bold', 'font-size': '16px'}),
        dcc.Dropdown(
            id='temporada-dropdown',
            options=[{'label': temporada, 'value': temporada} for temporada in df['temporada'].unique()],
            value=df['temporada'].iloc[0],  # Valor por defecto
            style={'width': '50%', 'margin': '20px auto'}
        ),
        dcc.Graph(id='grafico-precio-promedio', style={'height': '400px'}),
    ], style={'padding': '20px'}),  # Sección 2

    # Sección 3: Predicción de Precio Promedio
    html.Div([

        html.H3("Selecciona las características para el cálculo del Precio Promedio", style={'text-align': 'center', 'color': '#2f4f4f'}),

        # Dropdowns para seleccionar la Temporada (usando búsqueda)
        html.Div([
            html.Label("Selecciona una Temporada:", style={'font-weight': 'bold', 'font-size': '16px'}),
            dcc.Dropdown(
                id='dropdown-temporada',
                options=[{'label': temporada, 'value': temporada} for temporada in df['temporada'].unique()],
                placeholder="Selecciona una Temporada",
                searchable=True,  # Permite buscar dentro del dropdown
                clearable=True,   # Permite borrar la selección
            )
        ], style={'text-align': 'center', 'padding': '10px'}),

        # Dropdowns para seleccionar la Categoría de Producto (usando búsqueda)
        html.Div([
            html.Label("Selecciona una Categoría de Producto:", style={'font-weight': 'bold', 'font-size': '16px'}),
            dcc.Dropdown(
                id='dropdown-categoria',
                options=[{'label': categoria, 'value': categoria} for categoria in df['nombre_categoria_producto'].unique()],
                placeholder="Selecciona una Categoría",
                searchable=True,  # Permite buscar dentro del dropdown
                clearable=True,   # Permite borrar la selección
            )
        ], style={'text-align': 'center', 'padding': '10px'}),

        # Dropdowns para seleccionar la Ciudad Cliente (usando búsqueda)
        html.Div([
            html.Label("Selecciona una Ciudad Cliente:", style={'font-weight': 'bold', 'font-size': '16px'}),
            dcc.Dropdown(
                id='dropdown-ciudad',
                options=[{'label': ciudad, 'value': ciudad} for ciudad in df['ciudad_cliente'].unique()],
                placeholder="Selecciona una Ciudad Cliente",
                searchable=True,  # Permite buscar dentro del dropdown
                clearable=True,   # Permite borrar la selección
            )
        ], style={'text-align': 'center', 'padding': '10px'}),

        # Dropdowns para seleccionar el ID Producto (usando búsqueda)
        html.Div([
            html.Label("Selecciona un ID Producto:", style={'font-weight': 'bold', 'font-size': '16px'}),
            dcc.Dropdown(
                id='dropdown-idproducto',
                options=[{'label': str(id_producto), 'value': id_producto} for id_producto in df['id_producto'].unique()],
                placeholder="Selecciona un ID Producto",
                searchable=True,  # Permite buscar dentro del dropdown
                clearable=True,   # Permite borrar la selección
            )
        ], style={'text-align': 'center', 'padding': '10px'}),

        # Mostrar el resultado del precio promedio
        html.Div([
            html.H4("Precio Promedio Predicho:", id='precio-promedio', style={'text-align': 'center', 'color': '#2f4f4f'})
        ]),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'margin-top': '20px'})

])

# Callback para manejar la selección de los dropdowns y calcular el precio promedio
@app.callback(
    Output('precio-promedio', 'children'),
    [
        Input('dropdown-temporada', 'value'),
        Input('dropdown-categoria', 'value'),
        Input('dropdown-ciudad', 'value'),
        Input('dropdown-idproducto', 'value')
    ]
)
def calcular_precio_promedio(temporada_seleccionada, categoria_seleccionada, ciudad_seleccionada, id_producto_seleccionado):
    
    # Verificar si todas las selecciones están completas
    if not all([temporada_seleccionada, categoria_seleccionada, ciudad_seleccionada, id_producto_seleccionado]):
        return "Por favor, selecciona una opción para cada característica."
    
    # Filtrar los datos según la combinación seleccionada
    df_filtrado = df[
        (df['temporada'] == temporada_seleccionada) &
        (df['nombre_categoria_producto'] == categoria_seleccionada) &
        (df['ciudad_cliente'] == ciudad_seleccionada) &
        (df['id_producto'] == id_producto_seleccionado)
    ]
    
    # Comprobar si el DataFrame filtrado está vacío
    if df_filtrado.empty:
        return "No existe una combinación de datos para esta selección."
    
    # Calcular el precio promedio predicho
    precio_promedio = df_filtrado['predicted_price'].mean()
    
    # Comprobar si el precio promedio es NaN
    if pd.isna(precio_promedio):
        return "No se puede calcular el precio promedio, los datos son insuficientes o no válidos."
    
    return f"El precio promedio predicho es: {precio_promedio:.2f}"

# Callback para actualizar el gráfico de precio promedio por temporada
@app.callback(
    Output('grafico-precio-promedio', 'figure'),
    [Input('temporada-dropdown', 'value')]
)
def actualizar_grafico(temporada_seleccionada):
    df_filtrado = df[df['temporada'] == temporada_seleccionada]
    precio_promedio_temporada = df_filtrado.groupby('nombre_categoria_producto')['predicted_price'].mean().reset_index()
    
    return {
        'data': [
            {
                'x': precio_promedio_temporada['nombre_categoria_producto'],
                'y': precio_promedio_temporada['predicted_price'],
                'type': 'bar',
                'name': f'Precio Promedio - {temporada_seleccionada}',
            },
        ],
        'layout': {
            'title': f'Precio Promedio por Categoría en {temporada_seleccionada}',
            'xaxis': {'title': 'Categoría de Producto'},
            'yaxis': {'title': 'Precio Promedio'},
        },
    }

# Callback para actualizar la tabla de demanda según la temporada seleccionada
@app.callback(
    Output('tabla-demanda', 'data'),
    [Input('temporada-dropdown', 'value')]
)
def actualizar_tabla(temporada_seleccionada):
    df_filtrado = df[df['temporada'] == temporada_seleccionada]
    categoria_frecuencia = df_filtrado['nombre_categoria_producto'].value_counts().reset_index()
    categoria_frecuencia.columns = ['nombre_categoria_producto', 'frecuencia']
    categoria_frecuencia['demanda'] = categoria_frecuencia['frecuencia'].apply(
        lambda x: 'Alta' if x > 30 else ('Media' if x > 10 else 'Baja')
    )
    return categoria_frecuencia[['nombre_categoria_producto', 'demanda']].to_dict('records')

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8051, debug=True)

