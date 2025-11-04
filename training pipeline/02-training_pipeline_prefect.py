import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pathlib
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task
from mlflow import MlflowClient

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="Hyperparameter Tunning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv):
    
    mlflow.xgboost.autolog()
    
    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")
    
    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")
    
    train = xgb.DMatrix(X_train, label=y_train)
    
    valid = xgb.DMatrix(X_val, label=y_val)
    
    # ------------------------------------------------------------
    # Definir la función objetivo para Optuna
    #    - Recibe un `trial`, que se usa para proponer hiperparámetros.
    #    - Entrena un modelo con esos hiperparámetros.
    #    - Calcula la métrica de validación (RMSE) y la retorna (Optuna la minimizará).
    #    - Abrimos un run anidado de MLflow para registrar cada trial.
    # ------------------------------------------------------------
    def objective(trial: optuna.trial.Trial):
        # Hiperparámetros MUESTREADOS por Optuna en CADA trial.
        # Nota: usamos log=True para emular rangos log-uniformes (similar a loguniform).
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-3), 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha",   math.exp(-5), math.exp(-1), log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", math.exp(-6), math.exp(-1), log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", math.exp(-1), math.exp(3), log=True),
            "objective": "reg:squarederror",  
            "seed": 42,                      
        }

        # Run anidado para dejar rastro de cada trial en MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "xgboost")  # etiqueta informativa
            mlflow.log_params(params)                  # registra hiperparámetros del trial

            # Entrenamiento con early stopping en el conjunto de validación
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=10,
            )

            # Predicción y métrica en validación
            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Registrar la métrica principal
            mlflow.log_metric("rmse", rmse)

            # La "signature" describe la estructura esperada de entrada y salida del modelo:
            # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
            # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
            signature = infer_signature(X_val, y_pred)

            # Guardar el modelo del trial como artefacto en MLflow.
            mlflow.xgboost.log_model(
                booster,
                name="model",
                input_example=X_val[:5],
                signature=signature
            )

        # Optuna minimiza el valor retornado
        return rmse

    # ------------------------------------------------------------
    # Crear el estudio de Optuna
    #    - Usamos TPE (Tree-structured Parzen Estimator) como sampler.
    #    - direction="minimize" porque queremos minimizar el RMSE.
    # ------------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ------------------------------------------------------------
    # Ejecutar la optimización (n_trials = número de intentos)
    #    - Cada trial ejecuta la función objetivo con un set distinto de hiperparámetros.
    #    - Abrimos un run "padre" para agrupar toda la búsqueda.
    # ------------------------------------------------------------
    with mlflow.start_run(run_name="XGBoost Hyperparameter Optimization (Optuna)", nested=True):
        study.optimize(objective, n_trials=3)

    # --------------------------------------------------------
    # Recuperar y registrar los mejores hiperparámetros
    # --------------------------------------------------------
    best_params = study.best_params
    # Asegurar tipos/campos fijos (por claridad y consistencia)
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["seed"] = 42
    best_params["objective"] = "reg:squarederror"

    return best_params

@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="Best model ever"):
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(best_params)

        # Etiquetas del run "padre" (metadatos del experimento)
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        })

        # --------------------------------------------------------
        # 7) Entrenar un modelo FINAL con los mejores hiperparámetros
        #    (normalmente se haría sobre train+val o con CV; aquí mantenemos el patrón original)
        # --------------------------------------------------------
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=10,
        )

        # Evaluar y registrar la métrica final en validación
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # --------------------------------------------------------
        # 8) Guardar artefactos adicionales (p. ej. el preprocesador)
        # --------------------------------------------------------
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # La "signature" describe la estructura esperada de entrada y salida del modelo:
        # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
        # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
        # Si X_val es la matriz dispersa (scipy.sparse) salida de DictVectorizer:
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)

        # Para que las longitudes coincidan, usa el mismo slice en y_pred
        signature = infer_signature(input_example, y_val[:5])

        # Guardar el modelo del trial como artefacto en MLflow.
        mlflow.xgboost.log_model(
            booster,
            name="model",
            input_example=input_example,
            signature=signature,
        )
    return None

@task(name="Add best model to model registry")
def add_best_model_to_registry(experiment_name : str, model_name : str) -> None:

    
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"ERROR: Experimento '{experiment_name}' no encontrado.")
            return
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error al buscar el experimento: {e}")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"],
        output_format="list"
    )

# Obtener el mejor run
    if len(runs) > 0:
        best_run = runs[0]
        print("🏆 Champion Run encontrado:")
        print(f"Run ID: {best_run.info.run_id}")
        print(f"RMSE: {best_run.data.metrics['rmse']}")
        print(f"Params: {best_run.data.params}")
    else:
        print("⚠️ No se encontraron runs con métrica RMSE.")

    result = mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name=model_name
    )

    version = result.version
    print(f"Modelo registrado como Versión: {version}")

    print(f"Asignando alias 'champion' a la Versión {version}...")
    client.set_registered_model_alias(
        name=model_name,
        alias="champion_test", # El alias es sin el @
        version=version
    )


@flow(name="Main Flow")
def main_flow(year: int, month_train: str, month_val: str) -> None:
    """The main training pipeline"""
    
    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    load_dotenv(override=True)  # Carga las variables del archivo .env
    EXPERIMENT_NAME = "/Users/ivan.morales@iteso.mx/nyc-taxi-experiment-prefect"
    MODEL_REGISTRY_NAME = "workspace.default.nyc-taxi-model-prefect"

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # Hyper-parameter Tunning
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    
    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # Add best model to model registry
    add_best_model_to_registry(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_REGISTRY_NAME
    )

if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02")