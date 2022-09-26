import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from utils import read_model_pickle, read_preprocessor, read_reducer


class PPSUS_binding_energy_regressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs) -> None:

        models_filepath = kwargs["models_filepath"]
        models_number = (
            kwargs["models_number"] if "models_number" in kwargs.keys() else 1
        )
        reducer_file_name = (
            kwargs["reducer_file_name"]
            if "reducer_file_name" in kwargs.keys()
            else None
        )
        scaler_file_name = (
            kwargs["scaler_file_name"] if "scaler_file_name" in kwargs.keys() else None
        )

        self.models = read_model_pickle(
            files_path=models_filepath, ensemble_numbers=models_number
        )

        self.reducer = None
        if reducer_file_name:
            self.reducer = read_reducer(
                files_path=models_filepath, file_name=reducer_file_name
            )

        self.scaler = None
        if scaler_file_name:
            self.scaler = read_preprocessor(
                files_path=models_filepath, file_name=scaler_file_name
            )

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler:
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

        if self.reducer:
            X = self.reducer.transform(X)

        pd_columns = [f"column_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(data=X, columns=pd_columns)

        return X

    def get_predictions(self, X: pd.DataFrame) -> list:
        results = []
        for model in self.models:
            results.append(model.predict(X))
        return list(*zip(*np.mean(results, axis=0)))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.to_frame().T
        descriptions = X["description"].to_list()

        cols_to_drop = ["description", "packstat", "yhh_planarity"]
        X.drop(columns=cols_to_drop, inplace=True)

        X = self.preprocess_data(X)
        y = self.get_predictions(X)

        df_dict = {
            "description": descriptions,
            "predictions": y,
        }

        return df_dict
