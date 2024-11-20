# pages/prediction.py
import dash
from dash import html, dcc, callback, Output, Input
import pandas as pd

# Importar funciones desde data.py
from data import (
    get_df,
    get_modelo,
    get_categorias_ordenadas,
    get_temporadas_ordenadas,
    get_ciudades_ordenadas,
    get_ids_productos_ordenados
)

dash.register_page(__name__, path='/predicciones', name='Predicciones de Precio Medio Promedio')

def layout():
    # Obtener listas ordenadas para los dropdowns
    categorias_ordenadas = get_categorias_ordenadas()
    temporadas_ordenadas = get_temporadas_ordenadas()
    ciudades_ordenadas = get_ciudades_ordenadas()
    ids_productos_ordenados = get_ids_productos_ordenados()

    return html.Div([
        html.H1("Predicciones de Precio Medio Promedio", style={'text-align': 'center', 'color': '#2f4f4f'}),

        # Selección de características para el cálculo del precio promedio
        html.Div([
            html.H3("Selecciona las características para el cálculo del Precio Promedio", style={'text-align': 'center', 'color': '#2f4f4f'}),
            html.Div([
                html.Label("Selecciona una Temporada:"),
                dcc.Dropdown(
                    id='temporada-dropdown-predicciones',
                    options=[{'label': temporada, 'value': temporada} for temporada in temporadas_ordenadas],
                    placeholder="Temporada",
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Selecciona una Categoría de Producto:"),
                dcc.Dropdown(
                    id='categoria-dropdown-predicciones',
                    options=[{'label': categoria, 'value': categoria} for categoria in categorias_ordenadas],
                    placeholder="Categoría",
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Selecciona una Ciudad Cliente:"),
                dcc.Dropdown(
                    id='ciudad-dropdown-predicciones',
                    options=[{'label': ciudad, 'value': ciudad} for ciudad in ciudades_ordenadas],
                    placeholder="Ciudad",
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Selecciona un ID Producto:"),
                dcc.Dropdown(
                    id='producto-dropdown-predicciones',
                    options=[{'label': id_prod, 'value': id_prod} for id_prod in ids_productos_ordenados],
                    placeholder="ID Producto",
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '45%', 'display': 'inline-block'}),
        ], style={'text-align': 'center', 'padding': '20px'}),

        # Mostrar el precio promedio calculado
        html.Div(
            id='resultado-precio-promedio',
            style={
                'text-align': 'center',
                'font-size': '24px',
                'margin-top': '20px',
                'padding': '10px',
                'border': '1px solid #ddd',
                'border-radius': '10px',
                'width': '50%',
                'margin-left': 'auto',
                'margin-right': 'auto'
            }
        ),
    ])

# Función para calcular el precio promedio usando el modelo
def calcular_precio_promedio(temporada_seleccionada, categoria_seleccionada, ciudad_seleccionada, id_producto_seleccionado):
    # Verificar que todas las selecciones estén completas
    if not all([temporada_seleccionada, categoria_seleccionada, ciudad_seleccionada, id_producto_seleccionado]):
        return None, "Por favor, selecciona una opción para cada característica."

    df = get_df()
    modelo_regresion_lineal = get_modelo()

    # Obtener las características adicionales del producto a partir del dataframe
    df_producto = df[df['id_producto'] == id_producto_seleccionado]

    if df_producto.empty:
        return None, "No se encontraron datos para el ID de producto seleccionado."

    # Obtener los valores necesarios (usando el primer registro encontrado)
    peso_producto_g = df_producto['peso_producto_g'].iloc[0]
    largo_producto_cm = df_producto['largo_producto_cm'].iloc[0]
    altura_producto_cm = df_producto['altura_producto_cm'].iloc[0]
    ancho_producto_cm = df_producto['ancho_producto_cm'].iloc[0]
    estado_cliente = df_producto['estado_cliente'].iloc[0]

    # Crear un DataFrame con las características seleccionadas
    input_data = pd.DataFrame({
        'temporada': [temporada_seleccionada],
        'nombre_categoria_producto': [categoria_seleccionada],
        'peso_producto_g': [peso_producto_g],
        'largo_producto_cm': [largo_producto_cm],
        'altura_producto_cm': [altura_producto_cm],
        'ancho_producto_cm': [ancho_producto_cm],
        'ciudad_cliente': [ciudad_seleccionada],
        'estado_cliente': [estado_cliente],
        'id_producto': [id_producto_seleccionado]
    })

    # Predecir el precio usando el modelo cargado
    try:
        precio_predicho = modelo_regresion_lineal.predict(input_data)[0]
        return precio_predicho, None
    except Exception as e:
        return None, f"Error al predecir el precio: {str(e)}"

# Callback para actualizar el precio promedio
@callback(
    Output('resultado-precio-promedio', 'children'),
    [Input('temporada-dropdown-predicciones', 'value'),
     Input('categoria-dropdown-predicciones', 'value'),
     Input('ciudad-dropdown-predicciones', 'value'),
     Input('producto-dropdown-predicciones', 'value')],
    prevent_initial_call=True
)
def actualizar_precio(temporada_seleccionada, categoria_seleccionada, ciudad_seleccionada, id_producto_seleccionado):
    precio_promedio, mensaje_error = calcular_precio_promedio(
        temporada_seleccionada, categoria_seleccionada, ciudad_seleccionada, id_producto_seleccionado)

    if mensaje_error:
        resultado = mensaje_error
    else:
        resultado = f"El Precio Promedio Predicho es: ${precio_promedio:.2f}"

    return resultado
