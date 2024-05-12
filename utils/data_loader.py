# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, english_sentences, marathi_sentences):
        self.english_sentences = english_sentences
        self.marathi_sentences = marathi_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.marathi_sentences[idx]

def create_data_loader(english_sentences, marathi_sentences, batch_size, split_ratio=0.8):
    split_index = int(len(english_sentences) * split_ratio)
    train_english = english_sentences[:split_index]
    train_marathi = marathi_sentences[:split_index]
    valid_english = english_sentences[split_index:]
    valid_marathi = marathi_sentences[split_index:]

    train_dataset = TextDataset(train_english, train_marathi)
    valid_dataset = TextDataset(valid_english, valid_marathi)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
