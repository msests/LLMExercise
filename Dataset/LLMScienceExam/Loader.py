import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import sys

class LLMScienceExamDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def LLMScienceExamLoader() -> LLMScienceExamDataset:
    file = os.path.join(os.path.dirname(__file__), "6000_train_examples.csv")
    return LLMScienceExamDataset(file)
