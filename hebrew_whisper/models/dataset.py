import os
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, language_folder):
        self.data = self.load_data(language_folder)

    def load_data(self, folder):
        data = []
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    data.append(text)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]    