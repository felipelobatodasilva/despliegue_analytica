# pages/analysis.py
import dash
from dash import html, dcc, callback, Output, Input
import dash.dash_table as dash_table
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import pandas as pd
import time  # Importacion para usar temporizadores

# Importar funciones de data.py
from data import (
    get_df,
    get_categorias_ordenadas,
    get_temporadas_ordenadas,
    get_demanda_df,
    get_data_final
)

dash.register_page(__name__, path='/', name='Análisis de Ventas')

def layout():
    df = get_df()
    categorias_ordenadas = get_categorias_ordenadas()
    temporadas_ordenadas = get_temporadas_ordenadas()
    demanda_df = get_demanda_df()
    data_final = get_data_final()

    return html.Div([
        html.H1("Análisis de Ventas Olist Ecommerce", style={'text-align': 'center', 'color': '#2f4f4f'}),

        # línea con selector de temporada y calendario
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='temporada-dropdown',
                    options=[{'label': temporada, 'value': temporada} for temporada in temporadas_ordenadas],
                    placeholder="Temporada",
                    style={'width': '200px'}
                ),
            ]),
            html.Div([
                # Contenedor con estilo para rango de fechas
                html.Div([
                    html.Label("Seleccione un Intervalo de Fechas", style={'text-align': 'center', 'margin-bottom': '10px'}),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=df['fecha_pedido'].min(),
                        max_date_allowed=data_final,
                        initial_visible_month=data_final,
                        start_date=data_final - timedelta(days=7),
                        end_date=data_final,
                        display_format='DD/MM/YYYY',
                        month_format='MMMM, YYYY',
                        style={'width': '100%'}
                    ),
                ], style={
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'border-radius': '10px',
                    'width': '400px',
                    'margin-left': 'auto',
                    'margin-right': 'auto',
                    'text-align': 'center'
                }),
            ], style={'margin-left': 'auto'}),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'padding': '10px 50px'}),

        # Contenedor para tabla de demanda y gráfico barras
        html.Div([
            html.Div([
                html.H3("Predicción de Demanda por Categorías", style={'text-align': 'center', 'color': '#2f4f4f'}),
                dash_table.DataTable(
                    id='tabla-demanda',
                    columns=[{"name": i, "id": i} for i in demanda_df.columns],
                    data=demanda_df.to_dict('records'),
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
                    page_size=10  # Paginacion
                ),
            ], style={'width': '48%', 'padding': '10px'}),

            html.Div([
                html.H3("Top 10 Categorías de Mayor Demanda", style={'text-align': 'center', 'color': '#2f4f4f', 'font-weight': 'bold', 'margin-top': '0'}),
                dcc.Graph(id='grafico-proyeccion-ventas', style={'height': '350px'}),
            ], style={'width': '48%', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-around'}),

        # Nuevo contenedor para selector de categorías y mapas vecinos
        html.Div([
            # Div para o seletor de categoria e intervalo de preço
            html.Div([
                html.H3("Seleccione una Categoría", style={'text-align': 'center', 'color': '#2f4f4f'}),
                dcc.Dropdown(
                    id='categoria-dropdown',
                    options=[{'label': categoria, 'value': categoria} for categoria in categorias_ordenadas],
                    placeholder="Categoría",
                    style={'width': '80%', 'margin-top': '10px', 'margin': '0 auto'}
                ),
                html.Div(
                    id='rango-precio',
                    style={
                        'text-align': 'center',
                        'font-size': '24px',
                        'margin-top': '20px',
                        'padding': '10px',
                        'border': '1px solid #ddd',
                        'border-radius': '10px',
                        'width': '80%',
                        'margin-left': 'auto',
                        'margin-right': 'auto'
                    }
                ),
            ], style={'width': '48%', 'padding': '20px'}),

            # Div para el mapa
            html.Div([
                html.H3("Ubicación de los Clientes", style={'text-align': 'center', 'color': '#2f4f4f'}),
                dcc.Graph(
                    id='mapa-clientes',
                    figure=go.Figure(go.Scattermapbox()).update_layout(
                        mapbox_style="open-street-map",
                        mapbox_center={"lat": -14.235, "lon": -51.9253},
                        mapbox_zoom=2.5,
                        height=600
                    )
                )
            ], style={'width': '48%', 'padding': '20px'}),
        ], style={'display': 'flex', 'justify-content': 'space-around'}),
    ])

# Callback para atualizar el cuadro de proyeccion de ventas
@callback(
    Output('grafico-proyeccion-ventas', 'figure'),
    Input('temporada-dropdown', 'value'),
    prevent_initial_call=True
)
def actualizar_grafico(temporada_selecionada):
    start_time = time.time()
    print("Iniciando o callback 'actualizar_grafico'")

    if temporada_selecionada:
        df = get_df()  # Obtener el df 
        df_filtrado = df[df['temporada'] == temporada_selecionada]
        print(f"Tamanho do DataFrame filtrado: {df_filtrado.shape}")

        # top 10 categorias
        top_10_categorias = df_filtrado['nombre_categoria_producto'].value_counts().nlargest(10).index
        df_filtrado = df_filtrado[df_filtrado['nombre_categoria_producto'].isin(top_10_categorias)]

        # Agrupar los datos
        df_agrupado = df_filtrado.groupby('nombre_categoria_producto')['predicted_price'].sum().reset_index()

        fig = px.bar(
            df_agrupado,
            x='nombre_categoria_producto',
            y='predicted_price',
            labels={'nombre_categoria_producto': 'Categoría', 'predicted_price': 'Proyección de Ventas'},
            title=f'Proyección de ventas en {temporada_selecionada}'
        )
        fig.update_layout(xaxis={'title': {'text': 'Categoría', 'standoff': 20}, 'tickangle': -45},
                          yaxis={'title': {'text': 'Proyección de Ventas', 'standoff': 30}},
                          margin={'b': 150})

        end_time = time.time()
        print(f"Callback 'actualizar_grafico' concluído em {end_time - start_time:.2f} segundos")
        return fig

    end_time = time.time()
    print(f"Callback 'actualizar_grafico' concluído sem seleção em {end_time - start_time:.2f} segundos")
    return {}

# Callback para mostrar el intervalo de precio estimado
@callback(
    Output('rango-precio', 'children'),
    Input('categoria-dropdown', 'value'),
    prevent_initial_call=True
)
def mostrar_rango_precio(categoria_selecionada):
    start_time = time.time()
    print("Iniciando o callback 'mostrar_rango_precio'")

    if categoria_selecionada:
        df = get_df()
        df_filtrado = df[df['nombre_categoria_producto'] == categoria_selecionada]
        print(f"Tamanho do DataFrame filtrado por categoria: {df_filtrado.shape}")
        if not df_filtrado.empty:
            precio_min = df_filtrado['predicted_price'].min()
            precio_max = df_filtrado['predicted_price'].max()
            end_time = time.time()
            print(f"Callback 'mostrar_rango_precio' concluído em {end_time - start_time:.2f} segundos")
            return f"${precio_min:.2f} - ${precio_max:.2f}"
        else:
            end_time = time.time()
            print(f"Callback 'mostrar_rango_precio' concluído sem dados em {end_time - start_time:.2f} segundos")
            return "No hay datos para esta categoría"

    end_time = time.time()
    print(f"Callback 'mostrar_rango_precio' concluído sem seleção em {end_time - start_time:.2f} segundos")
    return "Seleccione una categoría para ver el rango de precio estimado."

# Callback para atualizar el mapa
@callback(
    Output('mapa-clientes', 'figure'),
    Input('temporada-dropdown', 'value'),
    prevent_initial_call=True
)
def atualizar_mapa(temporada_selecionada):
    start_time = time.time()
    print("Iniciando o callback 'atualizar_mapa'")

    df = get_df()
    if temporada_selecionada:
        df_filtrado = df[df['temporada'] == temporada_selecionada]
        print(f"Tamanho do DataFrame filtrado para o mapa: {df_filtrado.shape}")

        # Limitar el numero de puntos del mapa
        max_pontos = 1000  
        if len(df_filtrado) > max_pontos:
            df_filtrado = df_filtrado.sample(n=max_pontos, random_state=42)
            print(f"DataFrame reduzido para {max_pontos} pontos para o mapa")

        fig = px.scatter_mapbox(
            df_filtrado,
            lat="lat",
            lon="lon",
            hover_name="ciudad_cliente",
            color_discrete_sequence=["blue"],
            zoom=3,
            height=600
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": -14.235, "lon": -51.9253},
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        end_time = time.time()
        print(f"Callback 'atualizar_mapa' concluído em {end_time - start_time:.2f} segundos")
        return fig

    # Devolver mapa vacío si no seleccionan la temporada
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": -14.235, "lon": -51.9253},
        mapbox_zoom=2.5,
        height=600
    )
    end_time = time.time()
    print(f"Callback 'atualizar_mapa' concluído sem seleção em {end_time - start_time:.2f} segundos")
    return fig
