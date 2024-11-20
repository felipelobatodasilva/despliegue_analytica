# main.py
from dash import html, dcc
from dash_app import app
import dash

# Definir o layout da aplicação
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        # Menu de navegação
        dcc.Link('Análisis de Ventas', href='/'),
        html.Br(),
        dcc.Link('Predicciones de Precio Medio Promedio', href='/predicciones'),
    ], style={'padding': '20px'}),
    dash.page_container
])

if __name__ == '__main__':
    # Alterado para permitir acesso externo
    app.run_server(debug=True, host='0.0.0.0', port=8050)
