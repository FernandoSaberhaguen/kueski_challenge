# Aquí puedes poner funciones utilitarias que se usan en varios lugares del proyecto
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)
