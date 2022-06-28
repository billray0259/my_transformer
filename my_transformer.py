import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

class AttentionHead(nn.Module):

    def __init__(self, d_model):
        super(AttentionHead, self).__init__()
        self.key_layer = nn.Linear(d_model, d_model, bias=False)
        self.value_layer = nn.Linear(d_model, d_model, bias=False)
        self.query_layer = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scale_factor = 1/(d_model**0.5)
    
    def forward(self, x, mask):
        # x (batch_size, seq_len, d_model)
        # mask (batch_size, seq_len)
        keys = self.key_layer(x) # (batch_size, seq_len, d_model)
        queries = self.query_layer(x) # (batch_size, seq_len, d_model)

        attention_scores = keys @ queries.transpose(1, 2) * self.scale_factor # (batch_size, seq_len, seq_len)
        batch_idx, position_idx = torch.where(mask==0)
        attention_scores[batch_idx, :, position_idx] = -1e12 # broadcast over every position
        weights = self.softmax(attention_scores) # (batch_size, seq_len, seq_len)

        values = self.value_layer(x) # (batch_size, seq_len, d_model)

        hidden_state = weights @ values # (batch_size, seq_len, d_model)

        return hidden_state


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model=d_model) for _ in range(n_heads)])
        self.linear = nn.Linear(d_model * n_heads, d_model)
    
    def forward(self, x, mask):
        # x (batch_size, seq_length, d_model)
        z = [head(x, mask) for head in self.heads] # array of (batch_size, seq_length, d_model)
        z = torch.cat(z, dim=-1) # (batch_size, seq_length, d_model * n_heads)
        z = self.linear(z) # (batch_size, seq_length, d_model)

        return z


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads):
        super(TransformerBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        z = self.multi_head_attn(x, mask)
        z = self.layer_norm1(z)
        z = x + z
        middle = z
        z = self.linear1(z)
        z = self.relu(z)
        z = self.linear2(z)
        z = self.layer_norm2(z)
        z = middle + z
        return z


class PositionalEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, seq_length, padding_idx=None):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        self.embeddings = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.position_encodings = nn.Parameter(self._generate_position_encodings(), requires_grad=False)
    
    def forward(self, input_ids):
        z = self.embeddings(input_ids) # (batch_size, seq_length, d_model)
        z += self.position_encodings
        return z


    def _generate_position_encodings(self):
        i, pos = np.meshgrid(np.arange(self.d_model), np.arange(self.seq_length))
        encodings = np.sin(pos/10000**(i/self.d_model))
        encodings[:, 1::2] = np.cos(pos/10000**(i/self.d_model))[:, 1::2]
        encodings = torch.tensor(encodings).unsqueeze(0)
        return encodings


class TransformerBody(nn.Module):

    def __init__(self, n_layers, d_model, n_heads, output_layers=None):
        super(TransformerBody, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])

        self.output_layer_idx = output_layers if output_layers is not None else [-1] # layer ids of which hidden states should be output, [0] = output only the first hidden state [2, 3, 4] = output the 3rd 4th and 5th hidden states (output_layers indexes at 0)
        self.output_layer_idx = list(map(lambda id: id if id > 0 else n_layers + id, self.output_layer_idx))


    def forward(self, x, mask):
        z = x
        outputs = {}
        for i, layer in enumerate(self.layers):
            z = layer(z, mask)
            if i in self.output_layer_idx:
                outputs[i] = z

        if len(self.output_layer_idx) == 1:
            return outputs[self.output_layer_idx[0]]
        
        return outputs


class MyBERT(nn.Module):

    def __init__(self, vocab_size, seq_length, n_layers, d_model, n_heads, padding_idx=None):
        super(MyBERT, self).__init__()
        self.positional_embeddings = PositionalEmbedding(vocab_size, d_model, seq_length, padding_idx=padding_idx)
        self.transformer_body = TransformerBody(n_layers, d_model, n_heads)
        self.classification_head = nn.Linear(d_model, vocab_size)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        z = self.positional_embeddings(input_ids)
        z = self.transformer_body(z, attention_mask)
        logits = self.classification_head(z)

        if labels is not None:
            loss = self.calculate_loss(logits, labels)
            return (loss,)
        
        return (logits,)
    
    def calculate_loss(self, logits, labels):
        # logits (batch_size, seq_length, vocab_size)
        # labels (batch_size, seq_length)
        loss = self.cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss
