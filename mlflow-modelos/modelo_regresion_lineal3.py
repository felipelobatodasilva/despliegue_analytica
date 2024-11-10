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
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_parquet('../files_parquet/df_baseFinal.parquet', engine='pyarrow')
features = ['temporada', 'nombre_categoria_producto', 'peso_producto_g', 
            'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm', 
            'ciudad_cliente', 'estado_cliente', "id_producto"]
target = 'precio'
X = df[features]
y = df[target]

numerical_features = ['peso_producto_g', 'largo_producto_cm', 'altura_producto_cm', 'ancho_producto_cm']
categorical_features = ['temporada', 'nombre_categoria_producto', 'ciudad_cliente', 'estado_cliente', "id_producto"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression, k=5)),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

experiment_id = mlflow.set_experiment("modelo_regresion_lineal1").experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    # Treinamento e logging...
    # Defina seu código para treinar o modelo e registrar as métricas aqui
    
    # Exemplo:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.sklearn.log_model(model, "modelo_regresion_lineal1")

