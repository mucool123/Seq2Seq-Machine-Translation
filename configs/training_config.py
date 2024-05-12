# configs/training_config.py
config = {
    'd_model': 512,
    'ffn_hidden': 2048,
    'num_heads': 16,
    'drop_prob': 0.1,
    'num_layers': 2,
    'max_sequence_length': 180,
    'batch_size': 64,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'train_split': 0.8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_en': '/data/train.en',  # Update paths as necessary
    'train_mr': '/data/train.mr',
    'vocabulary_size': None,  # This will be set in the script
    'english_to_index': None,  # This will be populated from data
    'marathi_to_index': None,  # This will be populated from data
}
