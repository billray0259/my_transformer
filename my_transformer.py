from difflib import context_diff
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class AttentionHead(nn.Module):

    def __init__(self, config):
        super(AttentionHead, self).__init__()
        partial_size = config.hidden_size//config.num_attention_heads
        self.key_layer = nn.Linear(config.hidden_size, partial_size, bias=False)
        self.value_layer = nn.Linear(config.hidden_size, partial_size, bias=False)
        self.query_layer = nn.Linear(config.hidden_size, partial_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scale_factor = 1/(partial_size**0.5)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, x, mask):
        # x (batch_size, seq_len, d_model)
        # mask (batch_size, seq_len)
        keys = self.key_layer(x) # (batch_size, seq_len, d_model)
        queries = self.query_layer(x) # (batch_size, seq_len, d_model)

        attention_scores = keys @ queries.transpose(1, 2) * self.scale_factor # (batch_size, seq_len, seq_len)
        batch_idx, position_idx = torch.where(mask==0)
        attention_scores[batch_idx, :, position_idx] = -1e12 # broadcast over every position
        weights = self.softmax(attention_scores) # (batch_size, seq_len, seq_len)
        weights = self.dropout(weights)

        values = self.value_layer(x) # (batch_size, seq_len, d_model)

        hidden_state = weights @ values # (batch_size, seq_len, d_model)

        return hidden_state


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.num_attention_heads)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear((config.hidden_size//config.num_attention_heads) * config.num_attention_heads, config.hidden_size)
    
    def forward(self, x, mask):
        # x (batch_size, seq_length, d_model)
        z = [head(x, mask) for head in self.heads] # array of (batch_size, seq_length, d_model)
        z = torch.cat(z, dim=-1) # (batch_size, seq_length, d_model * n_heads)
        z = self.dropout(z)
        z = self.linear(z) # (batch_size, seq_length, d_model)

        return z
    

class MultiHeadAttention2(nn.Module):

    def __init__(self, config):
        super(MultiHeadAttention2, self).__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) must be a multiple of the number of attention heads (%d)" % (config.hidden_size, config.num_attention_heads))

        partial_size = config.hidden_size//config.num_attention_heads
        self.scale_factor = 1/partial_size**0.5

        weight_shape = (1, config.num_attention_heads, 3, config.hidden_size, partial_size)
        weight_tensor = torch.empty(weight_shape)
        self.weights = nn.Parameter(weight_tensor, requires_grad=True)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        # self.bias = nn.Parameter(torch.zeros(1, config.num_attention_heads, 3, config.hidden_size))

        self.softmax = nn.Softmax(dim=2)
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

    
    def forward(self, x, mask=None):
        kvq = x[:, None, None, :, :] @ self.weights # (batch_size, n_heads, 3, seq_len, partial_size)
        
        keys, values, queries = torch.unbind(kvq, dim=2) # 3 x (batch_size, n_heads, seq_len, partial_size)

        attention_scores = keys @ queries.transpose(-2, -1) * self.scale_factor # (batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            batch_idx, position_idx = torch.where(mask==0)
            attention_scores[batch_idx, :, :, position_idx] = -1e12 # broadcast over every position

        weights = self.softmax(attention_scores) # (batch_size, n_heads, seq_len, seq_len)
        weights = self.attn_dropout(weights)
        partial_hidden_states = weights @ values # (batch_size, n_heads, seq_len, partial_size)

        head_chunks = torch.unbind(partial_hidden_states, dim=1) # list of (batch_size, seq_len, partial_size)
        hidden_state = torch.cat(head_chunks, dim=2) # (batch_size, seq_len, d_model)

        hidden_state = self.dropout(hidden_state)
        hidden_state = self.linear(hidden_state) # (batch_size, seq_len, d_model)

        return hidden_state



class TransformerBlock(nn.Module):

    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention2(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)

        self.activation = config.hidden_act
        if type(self.activation) is str:
            if hasattr(nn, self.activation):
                self.activation = getattr(nn, self.activation)()
            elif hasattr(F, self.activation):
                self.activation = getattr(F, self.activation)
            else:
                raise KeyError("Unknown activation function: {}".format(self.activation))

        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x, mask=None):
        z = self.multi_head_attn(x, mask)
        z = self.dropout(z)
        z = x + z
        z = self.layer_norm1(z)
        middle = z
        # z = self.dropout(z)
        z = self.linear1(z)
        z = self.activation(z)
        # z = self.dropout(z)
        z = self.linear2(z)
        z = self.layer_norm2(z)
        z = middle + z
        return z


class PositionalEmbedding(nn.Module):

    def __init__(self, config):
        super(PositionalEmbedding, self).__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_encodings = nn.Parameter(self._generate_position_encodings(), requires_grad=False)
    
    def forward(self, input_ids):
        z = self.embeddings(input_ids) # (batch_size, seq_length, d_model)
        z += self.position_encodings
        return z


    def _generate_position_encodings(self):
        i, pos = np.meshgrid(np.arange(self.config.hidden_size), np.arange(self.config.max_position_embeddings))
        encodings = np.sin(pos/10000**(i/self.config.hidden_size))
        encodings[:, 1::2] = np.cos(pos/10000**(i/self.config.hidden_size))[:, 1::2]
        encodings = torch.tensor(encodings).unsqueeze(0)
        return encodings


class TransformerBody(nn.Module):

    def __init__(self, config):
        super(TransformerBody, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        # self.output_layer_idx = output_layers if output_layers is not None else [-1] # layer ids of which hidden states should be output, [0] = output only the first hidden state [2, 3, 4] = output the 3rd 4th and 5th hidden states (output_layers indexes at 0)
        # self.output_layer_idx = list(map(lambda id: id if id > 0 else n_layers + id, self.output_layer_idx))


    def forward(self, x, mask):
        z = x
        # outputs = {}
        # for i, layer in enumerate(self.layers):
        z = self.layers[0](z, mask)

        for layer in self.layers[1:]:
            z = layer(z)
            # if i in self.output_layer_idx:
            #     outputs[i] = z

        # if len(self.output_layer_idx) == 1:
        # return outputs[self.output_layer_idx[0]]
        return z
        
        # return outputs


class MyBert(nn.Module):

    def __init__(self, config):
        super(MyBert, self).__init__()
        self.positional_embeddings = PositionalEmbedding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.transformer_body = TransformerBody(config)
        self.classification_head = nn.Linear(config.hidden_size, config.vocab_size)

        # self.classification_head.weight = self.positional_embeddings.embeddings.weight
        # self.classification_bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # self.classification_head.weight = self.positional_embeddings.embeddings.weight.copy()
        # self.classification_head.weight.requires_grad = False

        z = self.positional_embeddings(input_ids)
        z = self.dropout(z)
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
