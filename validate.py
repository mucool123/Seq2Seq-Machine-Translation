import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.transformer import Transformer  # Ensure this path is correct
from configs.training_config import config
import numpy as np

# Define the Dataset class (if it's not imported from another file)
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, english_sentences, marathi_sentences, english_to_index, marathi_to_index):
        self.english_sentences = [torch.tensor([english_to_index.get(word, english_to_index['<PAD>']) for word in sent]) for sent in english_sentences]
        self.marathi_sentences = [torch.tensor([marathi_to_index.get(word, marathi_to_index['<PAD>']) for word in sent]) for sent in marathi_sentences]

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.marathi_sentences[idx]

# Load data function
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().lower() for line in file.readlines()]

# Load model function
def load_model(model_path, device):
    model = Transformer(
        d_model=config['d_model'],
        ffn_hidden=config['ffn_hidden'],
        num_heads=config['num_heads'],
        drop_prob=config['drop_prob'],
        num_layers=config['num_layers'],
        max_sequence_length=config['max_sequence_length'],
        vocab_size=config['vocab_size'],  # Ensure vocab size is defined in config
        device=device
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Validation function
def validate(model, loader, criterion, device):
    total_loss = 0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for eng, mar in loader:
            eng, mar = eng.to(device), mar.to(device)
            outputs = model(eng, mar[:, :-1])  # Assuming model takes input and target as separate arguments
            outputs = outputs.reshape(-1, outputs.shape[2])
            loss = criterion(outputs, mar[:, 1:].reshape(-1))
            total_loss += loss.item()
            total += len(mar)
            total_correct += (outputs.argmax(dim=1) == mar[:, 1:].reshape(-1)).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total
    return avg_loss, accuracy

# Main function
def main():
    device = torch.device(config['device'])
    english_sentences, marathi_sentences = load_data(config['train_mr']), load_data(config['train_en'])
    english_to_index = {word: i for i, word in enumerate(set(' '.join(english_sentences)))}
    marathi_to_index = {word: i for i, word in enumerate(set(' '.join(marathi_sentences)))}
    dataset = TextDataset(english_sentences, marathi_sentences, english_to_index, marathi_to_index)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = load_model('path_to_model_weights.pth', device)
    criterion = nn.CrossEntropyLoss()  # Adjust as necessary
    avg_loss, accuracy = validate(model, loader, criterion, device)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    main()
