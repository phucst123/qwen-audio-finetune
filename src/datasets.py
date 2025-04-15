import pandas as pd
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "id": row["id"],
            "url": row["url"],
            "description": row["description"],
        }
