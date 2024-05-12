import torch
from models.transformer import Transformer  # Ensure this path is correct
from configs.training_config import config
import numpy as np

# Function to load the trained Transformer model
def load_model(model_path, device):
    model = Transformer(
        d_model=config['d_model'],
        ffn_hidden=config['ffn_hidden'],
        num_heads=config['num_heads'],
        drop_prob=config['drop_prob'],
        num_layers=config['num_layers'],
        max_sequence_length=config['max_sequence_length'],
        vocab_size=config['vocab_size'],  # Make sure vocab size is defined in config
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Tokenize input sentence and prepare it for model input
def tokenize_sentence(sentence, english_to_index):
    tokens = [english_to_index.get(word, english_to_index['<PAD>']) for word in sentence.split()]
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

# Function to perform the translation
def translate_sentence(model, sentence, english_to_index, index_to_marathi, device):
    # Tokenizing input sentence
    tokens = [english_to_index.get(token, english_to_index[PADDING_TOKEN]) for token in sentence.lower().split()]
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_indices = output.argmax(2).squeeze(0).tolist()

    translated_sentence = ' '.join(index_to_marathi[idx] for idx in predicted_indices if idx in index_to_marathi)

    return translated_sentence

# Main function to execute translation
def main():
    device = torch.device(config['device'])
    model_path = 'path_to_your_saved_model.pth'
    model = load_model(model_path, device)
    
    # Example usage
    test_sentence = "What is your name?"
    translated_sentence = translate_sentence(model, test_sentence, config['english_to_index'], config['index_to_marathi'], device)
    print(f'Translated Sentence: {translated_sentence}')

if __name__ == '__main__':
    main()
