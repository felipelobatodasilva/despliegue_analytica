# Instalar pacotes no Databricks (se necessário)
# %pip install mlflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

# Definir o caminho do arquivo Parquet no Databricks File System (DBFS)
df = pd.read_parquet("/tmp/df_basefinal.parquet")

# Seleção de características e alvo
features = ['temporada', 'nombre_categoria_producto', 'peso_producto_g', 
            'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm', 
            'ciudad_cliente', 'estado_cliente', "id_producto"]
target = 'precio'

# Preparação dos dados
X = df[features]
y = df[target]

# Definir colunas numéricas e categóricas
numerical_features = ['peso_producto_g', 'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm']
categorical_features = ['temporada', 'nombre_categoria_producto', 'ciudad_cliente', 'estado_cliente', "id_producto"]

# Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Configuração do modelo com pipeline usando regressão linear
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar o MLflow para registrar o experimento
with mlflow.start_run() as run:
    # Treinar o modelo
    model.fit(X_train, y_train)

    # Predições
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cross_val_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())

    # Logar as métricas no MLflow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Cross-validated RMSE", cross_val_rmse)
    
    # Logar o modelo no MLflow
    mlflow.sklearn.log_model(model, "LinearRegressionModel")

    # Exibir as métricas
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R²: {r2}")
    print(f"Cross-validated RMSE: {cross_val_rmse}")

    # Predição para um novo dado
    new_data = pd.DataFrame({
        'temporada': ['Invierno'],
        'nombre_categoria_producto': ['cama_mesa_banho'],
        'id_producto': ['364e789259da982f5b7e43aaea7be61'],
        'peso_producto_g': [750],
        'largo_producto_cm': [16],
        'altura_producto_cm': [10],
        'ancho_producto_cm': [16],
        'ciudad_cliente': ['sao_paulo'],
        'estado_cliente': ['SP']
    })
    predicted_price = model.predict(new_data)
    print(f"Predicted Seasonal Price: {predicted_price[0]}")

    # Logar a predição para referência futura
    mlflow.log_param("Predicted Seasonal Price", predicted_price[0])
