import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pandas as pd
import xgboost as xgb # Aunque no se usa, evitar error de importación en MLflow
import warnings

from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from prefect import flow, task

# --- Constantes del Flow ---
EXPERIMENT_NAME = "/Users/ivan.morales@iteso.mx/nyc-taxi-experiment-prefect"
MODEL_REGISTRY_NAME = "workspace.default.nyc-taxi-model-prefect"

# --- Task 1: Cargar y Preprocesar Datos---
@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Lee un Parquet, calcula la duración y filtra columnas."""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df

# --- Task 2: Ingeniería de Features (de Tarea 5) ---
@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Crea el DictVectorizer. 
    NOTA: Basado en Tarea 5, usamos ['PULocationID', 'DOLocationID'] 
    como features, no 'PU_DO'.
    """
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    dv = DictVectorizer()
    
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    
    print(f"Vocabulario del DV (features): {len(dv.feature_names_)}")
    return X_train, X_val, y_train, y_val, dv

# --- Task 3: Entrenar Challenger 1: Gradient Boosting ---
@task(name="Train Challenger: Gradient Boosting")
def train_gradient_boosting(X_train, X_val, y_train, y_val, dv):
    """Entrena, optimiza y registra un HistGradientBoostingRegressor."""
    print("Iniciando entrenamiento de Gradient Boosting...")
    
    # Ignorar warnings de `toarray()` que pueden ser grandes
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    def objective(trial: optuna.trial.Trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-3), 1.0, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", math.exp(-6), math.exp(-1), log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "sklearn-hist-gradient-boosting")
            mlflow.log_params(params)
            model = HistGradientBoostingRegressor(
                **params,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                loss='squared_error'
            )
            
            # Convertir a denso (requerido por HGBR)
            X_train_dense = X_train.toarray()
            X_val_dense = X_val.toarray()
            
            model.fit(X_train_dense, y_train)
            y_pred = model.predict(X_val_dense)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            # Loggear modelo
            signature = infer_signature(X_val_dense, y_pred)
            mlflow.sklearn.log_model(
                model, "model",
                input_example=X_val_dense[:5],
                signature=signature
            )
        return rmse

    # Iniciar el estudio de Optuna
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Iniciar el Run PADRE de MLflow
    with mlflow.start_run(run_name="Challenger GB Optuna (Prefect)") as parent_run:
        mlflow.set_tag("model_type", "GradientBoosting")
        study.optimize(objective, n_trials=10) # 10 trials como en Tarea 5
        
        # Entrenar modelo final con los mejores params
        best_params = study.best_params
        best_params["max_depth"] = int(best_params["max_depth"])
        
        final_model = HistGradientBoostingRegressor(
            **best_params,
            max_iter=100, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=42, loss='squared_error'
        )
        
        X_train_dense = X_train.toarray()
        X_val_dense = X_val.toarray()
        final_model.fit(X_train_dense, y_train)
        
        y_pred = final_model.predict(X_val_dense)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        
        # Loggear artefacto de preprocesador
        with open("preprocessor_gb.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor_gb.pkl", artifact_path="preprocessor")

        # Loggear modelo final
        signature = infer_signature(X_val_dense, y_pred)
        mlflow.sklearn.log_model(
            final_model, "model",
            input_example=X_val_dense[:5],
            signature=signature
        )
        
        print(f"Gradient Boosting entrenado. RMSE: {rmse:.4f}")
        return parent_run.info.run_id, rmse

# --- Task 4: Entrenar Challenger 2: Random Forest ---
@task(name="Train Challenger: Random Forest")
def train_random_forest(X_train, X_val, y_train, y_val, dv):
    """Entrena, optimiza y registra un RandomForestRegressor."""
    print("Iniciando entrenamiento de Random Forest...")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    def objective(trial: optuna.trial.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "sklearn-rf")
            mlflow.log_params(params)
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            
            # RF Acepta sparse, no necesita .toarray() para entrenar
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            
            # Loggear modelo
            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(
                model, "model",
                input_example=X_val[:5],
                signature=signature
            )
        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name="Challenger RF Optuna (Prefect)") as parent_run:
        mlflow.set_tag("model_type", "RandomForest")
        study.optimize(objective, n_trials=10) # 10 trials como en Tarea 5
        
        best_params = study.best_params
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["min_samples_split"] = int(best_params["min_samples_split"])
        best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
        
        final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train)
        
        y_pred = final_model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        
        with open("preprocessor_rf.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor_rf.pkl", artifact_path="preprocessor")
        
        # Signature para el modelo final (usando denso para legibilidad)
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names) 
        signature = infer_signature(input_example, y_val[:5])
        
        mlflow.sklearn.log_model(
            final_model, "model",
            input_example=input_example,
            signature=signature
        )
        
        print(f"Random Forest entrenado. RMSE: {rmse:.4f}")
        return parent_run.info.run_id, rmse

# --- Task 5: Registrar al mejor challenger ---
@task(name="Register Best Challenger")
def register_best_challenger(gb_result, rf_result, model_name: str):
    """
    Compara los dos modelos entrenados y registra al mejor 
    con el alias '@challenger'.
    """
    client = MlflowClient()
    gb_run_id, gb_rmse = gb_result
    rf_run_id, rf_rmse = rf_result
    
    if gb_rmse < rf_rmse:
        print(f"Mejor retador: Gradient Boosting (RMSE: {gb_rmse:.4f})")
        winner_run_id = gb_run_id
        winner_rmse = gb_rmse
    else:
        print(f"Mejor retador: Random Forest (RMSE: {rf_rmse:.4f})")
        winner_run_id = rf_run_id
        winner_rmse = rf_rmse

    model_uri = f"runs:/{winner_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    challenger_version = mv.version
    print(f"Modelo registrado como Versión {challenger_version}.")
    
    client.set_registered_model_alias(
        name=model_name,
        alias="challenger",
        version=challenger_version
    )
    print(f"Alias '@challenger' asignado a la Versión {challenger_version}.")
    
    return challenger_version, winner_rmse

# --- Task 6: Promover al nuevo Campeón ---
@task(name="Promote Champion")
def promote_champion(challenger_version, challenger_rmse, X_val, y_val, dv, model_name: str):
    """
    Compara el '@challenger' contra el '@champion' existente.
    Promueve al mejor para que sea el nuevo '@champion'.
    """
    client = MlflowClient()
    
    # 1. Preparar datos de evaluación
    # (Necesario para que pyfunc.load_model funcione con la signature)
    feature_names = dv.get_feature_names_out()
    X_val_df = pd.DataFrame(X_val.toarray(), columns=feature_names)
    
    # 2. Obtener y evaluar al campeón actual
    try:
        champion_details = client.get_model_version_by_alias(model_name, "champion")
        champion_version = champion_details.version
        champion_uri = f"models:/{model_name}@champion"
        
        print(f"Evaluando al @champion actual (Versión {champion_version})...")
        champion_model = mlflow.pyfunc.load_model(champion_uri)
        y_pred_champ = champion_model.predict(X_val_df)
        champion_rmse = root_mean_squared_error(y_val, y_pred_champ)
        print(f"RMSE del @champion (Versión {champion_version}): {champion_rmse:.4f}")

    except MlflowException as e:
        print("No se encontró un @champion existente. El retador será promovido por defecto.")
        champion_rmse = float('inf')
        champion_version = "N/A"

    # 3. Comparar y promover
    if challenger_rmse < champion_rmse:
        print(f"¡NUEVO CAMPEÓN!")
        print(f"El retador (Versión {challenger_version}, RMSE: {challenger_rmse:.4f}) "
              f"supera al campeón (Versión {champion_version}, RMSE: {champion_rmse:.4f}).")
        
        # Asignar '@champion' a la nueva versión
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=challenger_version
        )
    else:
        print(f"¡EL CAMPEÓN SE MANTIENE!")
        print(f"El campeón (Versión {champion_version}, RMSE: {champion_rmse:.4f}) "
              f"sigue siendo mejor que el retador (Versión {challenger_version}, RMSE: {challenger_rmse:.4f}).")

    # 4. Limpiar alias '@challenger'
    print(f"Limpiando alias '@challenger' de la Versión {challenger_version}.")
    client.delete_registered_model_alias(
        name=model_name,
        alias="challenger"
    )

# --- main flow ---
@flow(name="Train Challenger Flow") # Requisito 6: Nombre único
def main_challenger_flow(
    train_path: str = '../data/green_tripdata_2025-01.parquet', # Enero
    val_path: str = '../data/green_tripdata_2025-02.parquet'   # Febrero
):
    """
    Flow para entrenar y comparar dos modelos retadores (GB y RF) 
    y promover al mejor contra el campeón existente.
    """
    
    # Cargar .env y configurar MLflow
    load_dotenv(override=True)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc") # Usar Unity Catalog
    
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    
    # 1. Cargar datos
    # Usamos Enero y Febrero para entrenar
    df_train_jan = read_data(train_path)
    df_train_feb = read_data(val_path)
    df_train = pd.concat([df_train_jan, df_train_feb])
    
    # Usamos Marzo para validar
    df_val = read_data('../data/green_tripdata_2025-03.parquet')

    # 2. Ingeniería de Features
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # 3. Entrenar ambos challengers en paralelo
    gb_result = train_gradient_boosting.submit(X_train, X_val, y_train, y_val, dv)
    rf_result = train_random_forest.submit(X_train, X_val, y_train, y_val, dv)

    # 4. Registrar al mejor challenger
    challenger_version, challenger_rmse = register_best_challenger(
        gb_result=gb_result.result(), 
        rf_result=rf_result.result(), 
        model_name=MODEL_REGISTRY_NAME
    )
    
    # 5. Comparar y promover al campeón
    promote_champion(
        challenger_version,
        challenger_rmse,
        X_val, y_val, dv,
        model_name=MODEL_REGISTRY_NAME
    )

if __name__ == "__main__":
    main_challenger_flow(
        train_path='../data/green_tripdata_2025-01.parquet',
        val_path='../data/green_tripdata_2025-02.parquet'
    )