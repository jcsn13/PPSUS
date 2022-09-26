import pickle
import umap.umap_ as umap
from tensorflow.keras.models import load_model
from keras.engine.sequential import Sequential
from sklearn.base import TransformerMixin
from typing import List


def read_model_pickle(files_path: str, ensemble_numbers: int = 1) -> List[Sequential]:
    models = []
    for i in range(ensemble_numbers):
        models.append(load_model(f"{files_path}/model_{i}"))
    return models


def read_preprocessor(files_path: str, file_name: str) -> TransformerMixin:
    scaler = pickle.load(open(f"{files_path}/{file_name}", "rb"))
    return scaler


def read_reducer(files_path: str, file_name: str) -> umap.UMAP:
    reducer = pickle.load(open(f"{files_path}/{file_name}", "rb"))
    return reducer
