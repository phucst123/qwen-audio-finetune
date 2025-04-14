import pandas as pd
from torch.utils.data import DataLoader


def load_datasets(train_path, val_path):
    """Load training and validation datasets."""
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"Loaded {len(df_train)} training examples")
    print(f"Loaded {len(df_val)} validation examples")

    return df_train, df_val


def create_dataloader(dataset, data_collator, batch_size, shuffle=True):
    """Create a dataloader for the dataset."""
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle
    )
