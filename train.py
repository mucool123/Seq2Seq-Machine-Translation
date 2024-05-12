import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from models.transformer import Transformer  # Ensure this path is correct
from configs.training_config import config  # Import the configuration
import numpy as np

# Add this in training_config.py or at the beginning of train.py
START_TOKEN = '<START>'
PADDING_TOKEN = '<PAD>'  # Changed to '<PAD>' for consistency
END_TOKEN = '<END>'

marathi_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ',
                      'ँ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ए', 'ऐ', 'ओ', 'औ',
                      'क', 'ख', 'ग', 'घ', 'ङ',
                      'च', 'छ', 'ज', 'झ', 'ञ',
                      'ट', 'ठ', 'ड', 'ढ', 'ण',
                      'त', 'थ', 'द', 'ध', 'न',
                      'प', 'फ', 'ब', 'भ', 'म',
                      'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
                      'क्ष', 'त्र', 'ज्ञ',
                      'श्र', PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

marathi_to_index = {ch: idx for idx, ch in enumerate(marathi_vocabulary)}
english_to_index = {ch: idx for idx, ch in enumerate(english_vocabulary)}
index_to_marathi = {idx: ch for idx, ch in enumerate(marathi_vocabulary)}


# Define the Dataset
class TextDataset(Dataset):
    def __init__(self, english_sentences, marathi_sentences):
        self.english_sentences = english_sentences
        self.marathi_sentences = marathi_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.marathi_sentences[idx]

# Load and prepare data
def load_data():
    def load_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]

    english_sentences = load_file(config['train_en'])
    marathi_sentences = load_file(config['train_mr'])

    return english_sentences, marathi_sentences

# Create DataLoaders
def create_datasets(english_sentences, marathi_sentences):
    dataset = TextDataset(english_sentences, marathi_sentences)
    train_size = int(len(dataset) * config['train_split'])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, valid_loader

# Training and validation function
def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for eng, mar in train_loader:
            eng, mar = eng.to(device), mar.to(device)
            optimizer.zero_grad()
            output = model(eng, mar)
            loss = criterion(output, mar)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for eng, mar in valid_loader:
                eng, mar = eng.to(device), mar.to(device)
                output = model(eng, mar)
                loss = criterion(output, mar)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')

    return train_losses, valid_losses

# Plotting function for losses
def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to execute training process
def main():
    device = torch.device(config['device'])
    english_sentences, marathi_sentences = load_data()
    train_loader, valid_loader = create_datasets(english_sentences, marathi_sentences)

    model = Transformer(
        d_model=config['d_model'],
        ffn_hidden=config['ffn_hidden'],
        num_heads=config['num_heads'],
        drop_prob=config['drop_prob'],
        num_layers=config['num_layers'],
        max_sequence_length=config['max_sequence_length'],
        vocab_size=config['vocab_size'],  # Ensure vocab size is defined in config
        device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_losses, valid_losses = train_and_validate(model, train_loader, valid_loader, criterion, optimizer, device, config['num_epochs'])
    plot_losses(train_losses, valid_losses)

if __name__ == '__main__':
    main()
