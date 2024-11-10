import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Cargar los datos
df = pd.read_parquet('../files_parquet/df_baseFinal.parquet')

# Selección de características y objetivo
features = ['temporada', 'nombre_categoria_producto', 'peso_producto_g',
            'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm',
            'ciudad_cliente', 'estado_cliente', 'id_producto']
target = 'precio'

# Preparación de los datos
X = df[features]
y = df[target]

# Definir columnas numéricas y categóricas
numerical_features = ['peso_producto_g', 'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm']
categorical_features = ['temporada', 'nombre_categoria_producto', 'ciudad_cliente', 'estado_cliente', 'id_producto']

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Configuración del modelo con pipeline usando regresión lineal
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar el experimento en MLflow
experiment_id = mlflow.set_experiment("modelo_regresion_lineal1").experiment_id

# Iniciar la ejecución en MLflow
with mlflow.start_run(experiment_id=experiment_id) as run:
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

    # Registrar métricas en MLflow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Cross-validated RMSE", cross_val_rmse)

    mlflow.sklearn.log_model(model, "modelo_regresion_lineal1")

    # Mostrar las métricas
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R²: {r2}")