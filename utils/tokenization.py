# utils/tokenization.py
import torch
import numpy as np

def tokenize(sentence, vocabulary, max_sequence_length, start_token='<START>', end_token='<END>', padding_token='<PADDING>'):
    """ Convert sentence to a list of tokens based on a vocabulary. """
    tokens = [vocabulary.get(word, vocabulary[padding_token]) for word in sentence.split()]
    tokens = [vocabulary[start_token]] + tokens + [vocabulary[end_token]]
    if len(tokens) < max_sequence_length:
        tokens += [vocabulary[padding_token]] * (max_sequence_length - len(tokens))
    return tokens[:max_sequence_length]

def create_masks(batch, max_sequence_length):
    """ Create padding masks for sequences in a batch. """
    batch_size = len(batch)
    mask = torch.zeros(batch_size, max_sequence_length, max_sequence_length, dtype=torch.bool)

    for idx, seq in enumerate(batch):
        seq_len = len([i for i in seq if i != 0])  # Assuming padding token index is 0
        if seq_len < max_sequence_length:
            mask[idx, :, seq_len:] = True
            mask[idx, seq_len:, :] = True

    return mask
