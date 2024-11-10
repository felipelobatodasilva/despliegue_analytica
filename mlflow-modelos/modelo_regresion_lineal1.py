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

mlflow.set_experiment("modelo1_regresion_lineal")

# Carregar dados
df = pd.read_parquet('../files_parquet/df_baseFinal.parquet', engine='pyarrow')

# Seleção de características e alvo
features = ['temporada', 'nombre_categoria_producto', 'peso_producto_g', 
            'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm', 
            'ciudad_cliente', 'estado_cliente', "id_producto"]
target = 'precio'
X = df[features]
y = df[target]

# Pré-processamento
numerical_features = ['peso_producto_g', 'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm']
categorical_features = ['temporada', 'nombre_categoria_producto', 'ciudad_cliente', 'estado_cliente', "id_producto"]

preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cross_val_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Cross-validated RMSE", cross_val_rmse)

    mlflow.sklearn.log_model(model, "LinearRegressionModel")

    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}, Cross-validated RMSE: {cross_val_rmse}")
