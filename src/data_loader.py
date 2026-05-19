import pandas as pd
from pathlib import Path


def load_data_raw(file_name: str):
    current_dir = Path(__file__).parent
    data_path = current_dir / '..' / "data" / "raw" / "wine+quality" / file_name
    data = pd.read_csv(data_path, sep=';')
    return data
