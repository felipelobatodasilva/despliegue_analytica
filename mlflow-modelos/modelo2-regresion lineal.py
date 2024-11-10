import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Cargar datos
df = pd.read_parquet('../files_parquet/df_baseFinal.parquet', engine='pyarrow')

# Selección de características y objetivo
features = ['temporada', 'nombre_categoria_producto', 'peso_producto_g', 
            'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm', 
            'ciudad_cliente', 'estado_cliente', "id_producto"]
target = 'precio'

X = df[features]
y = df[target]

# Preprocesamiento
numerical_features = ['peso_producto_g', 'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm']
categorical_features = ['temporada', 'nombre_categoria_producto', 'ciudad_cliente', 'estado_cliente', "id_producto"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Configuración del modelo con Regresión Lineal con normalize=True
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression(normalize=True))
])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar MLflow para registrar el experimento
with mlflow.start_run() as run:
    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cross_val_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())

    # Registrar las métricas en MLflow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Cross-validated RMSE", cross_val_rmse)
    
    # Registrar el modelo en MLflow
    mlflow.sklearn.log_model(model, "LinearRegressionModel_NormalizeTrue")

    # Mostrar las métricas
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R²: {r2}")
    print(f"Cross-validated RMSE: {cross_val_rmse}")

    # Predicción para un nuevo dato
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

    # Registrar la predicción en MLflow para referencia futura
    mlflow.log_param("Predicted Seasonal Price", predicted_price[0])