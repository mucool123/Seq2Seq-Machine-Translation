import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    """Return the best available device (GPU if available, otherwise CPU)."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def scaled_dot_product(q, k, v, mask=None):
    """Calculate the scaled dot product of queries, keys and values, considering an optional mask."""
    d_k = q.size()[-1]
    scale_factor = math.sqrt(d_k)
    if scale_factor == 0:
        raise ValueError("Division by zero error due to zero size dimension")
    
    scaled = torch.matmul(q, k.transpose(-1, -2)) / scale_factor
    if mask is not None:
        scaled += mask  # Make sure the mask is broadcastable to the shape of scaled
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model should be even for positional encoding.")

        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.register_buffer('pe', self._generate_encoding(max_sequence_length, d_model))

    def _generate_encoding(self, max_len, dimension):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        pe = torch.zeros(max_len, dimension)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SentenceEmbedding(nn.Module):
    """Create an embedding for a given sentence with optional positional encoding."""
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token, end_token):
        """Tokenizes a batch of sentences adding start and end tokens and applying padding."""
        tokenized_batch = []
        for sentence in batch:
            tokenized_sentence = [self.language_to_index.get(token, self.language_to_index[self.PADDING_TOKEN]) for token in sentence]
            if start_token:
                tokenized_sentence.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                tokenized_sentence.append(self.language_to_index[self.END_TOKEN])
            tokenized_sentence.extend([self.language_to_index[self.PADDING_TOKEN]] * (self.max_sequence_length - len(tokenized_sentence)))
            tokenized_batch.append(torch.tensor(tokenized_sentence[:self.max_sequence_length]))
        return torch.stack(tokenized_batch).to(get_device())

    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        x = self.position_encoder(x)
        return self.dropout(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs.
    """
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True) + self.eps
        normalized = (inputs - mean) / std
        return self.gamma * normalized + self.beta


  
class PositionwiseFeedForward(nn.Module):
    """
    Implements position-wise feedforward sublayer.
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class EncoderLayer(nn.Module):
    """
    Represents one layer of the transformer encoder.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(drop_prob)
        self.positionwise_feed_forward = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, self_attention_mask):
        attention_output = self.multi_head_attention(x, self_attention_mask)
        x = self.norm1(x + self.dropout1(attention_output))
        ffn_output = self.positionwise_feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x
    
class SequentialEncoder(nn.Sequential):
    """
    A sequential container for encoder layers.
    """
    def forward(self, x, self_attention_mask):
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    """
    Cross attention used in the transformer's decoder.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for equal distribution")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, seq_length, _ = x.size()
        kv = self.kv_layer(x).view(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
        q = self.q_layer(y).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k, v = kv.chunk(2, dim=-1)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_length, self.d_model)
        return self.final_linear(context)



class DecoderLayer(nn.Module):
    """Defines a single layer in the transformer's decoder stack."""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.layer_norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.layer_norm3 = LayerNormalization([d_model])
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        attention_output = self.self_attention(y, self_attention_mask)
        y = self.layer_norm1(attention_output + y)
        y = self.dropout1(y)

        attention_output = self.encoder_decoder_attention(x, y, cross_attention_mask)
        y = self.layer_norm2(attention_output + y)
        y = self.dropout2(y)

        ffn_output = self.ffn(y)
        y = self.layer_norm3(ffn_output + y)
        y = self.dropout3(y)
        return y



class SequentialDecoder(nn.Sequential):
    """Sequential container for handling multiple DecoderLayers."""
    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        for layer in self:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    """The Decoder part of the Transformer."""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        return self.layers(x, y, self_attention_mask, cross_attention_mask)


class Transformer(nn.Module):
    """Defines the Transformer model that combines the Encoder and Decoder components."""
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, vocab_size, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        enc_output = self.encoder(src, src_mask, src_padding_mask)
        dec_output = self.decoder(enc_output, tgt, tgt_mask, tgt_padding_mask)
        return self.linear(dec_output)