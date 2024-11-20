# dash_app.py
import dash

# Inicializa a aplicação Dash com suporte a múltiplas páginas
app = dash.Dash(__name__, suppress_callback_exceptions=True, use_pages=True)
server = app.server

# Configuração do cache (opcional, se precisar usar o cache em outros lugares)
# from flask_caching import Cache
# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'SimpleCache',
#     'CACHE_DEFAULT_TIMEOUT': 300  # Tempo em segundos
# })
