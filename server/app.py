import pandas as pd
import os
import time
from fastapi import FastAPI, UploadFile
from model import PPSUS_binding_energy_regressor

regressor = PPSUS_binding_energy_regressor(
    models_filepath="pickles",
    models_number=1,
    reducer_file_name="umap",
    scaler_file_name="scaler",
)

app = FastAPI()


@app.post("/predictions")
async def root(file: UploadFile) -> dict:
    csv_name = file.filename
    csv_path = ""
    file_path = os.path.join(csv_path, csv_name)
    with open(file_path, mode="wb+") as f:
        f.write(file.file.read())

    data = {}
    with open(file_path, mode="r", encoding="utf-8") as csvf:
        df = pd.read_csv(csvf)

        df["predictions"] = df.apply(regressor.predict, axis=1)
        predictions_dict = pd.json_normalize(df["predictions"])
        pred_size = df.shape[0]

        data = {
            str(element): {
                "complex_name": predictions_dict["description"][element][0],
                "predicted_value": float(predictions_dict["predictions"][element][0]),
            }
            for element in range(pred_size)
        }
    return data


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time:0.4f} sec")
    return response
