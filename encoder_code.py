import torch
import math
from torch import nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    # q, k, v = 30 x 8 x 200 x 64
    d_k = q.size()[-1] #64   pre cursor to self attention matrix (batch_size   num_heads    max_seq_len   max_seq_len)
    scaled = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(d_k) #    30     x    8      x      200      x   200
    if mask is not None:
        scaled += mask # before 200 x 200::after 30 x 8 x 200 x 200 matrix is adding to every batch and head, so just the last N dimensions need to match
    attention = F.softmax(scaled, dim=-1) # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v) # 30 x 8 x 200 x 64 :: every_batch x every_head x every_word x 64 dimensional embedding i.e values tensor 
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model//num_heads # 64
        self.qkv_layer = nn.Linear(d_model, 3*d_model) # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 x 512
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512 
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        q, k, v, = qkv.chunk(3, dim=-1) # each are 30 x 8 x 200 x 64 (192/3)
        values, attention = scaled_dot_product(q, k, v, mask)  ## attention = 30 x 8 x 200 x 200 & values = 30 x 8 x 200 x 64
        values = values.reshape(batch_size, sequence_length, self.num_heads*self.head_dim) # 30 x 200 x 512(8*64)
        out = self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # [512]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # [512]
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # [512]
        
    def forward(self, inputs): # 30 x 200 x 512
        dims = [-(i+1) for i in range(len(self.parameters_shape))] # [-1]
        mean = inputs.mean(dim = dims, keepdim=True) # 30 x 200 x 1
        var = ((inputs-mean)**2).mean(dim=dims, keepdim=True) # 30 x 200 x 1
        std = (var+self.eps).sqrt() # 30 x 200 x 1
        y = (inputs-mean)/std # 30 x 200 x 512
        out = self.gamma*y+self.beta # 30 x 200 x 512
        return out
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 x 2048
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x 512
        self.relu = nn.ReLu() # activation function which is used to helped the neural network to learn more complex patterns
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x): # 30 x 200 x 512
        x = self.linear1(x) # 30 x 200 x 2048
        x = self.relu(x) # 30 x 200 x 2048
        x = self.dropout(x) # 30 x 200 x 2048
        x = self.linear2(x) # 30 x 200 x 512
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.fnn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x # 30 x 200 x 512
        x = self.attention(x, mask=None) # 30 x 200 x 512
        x = self.dropout1(x) # 30 x 200 x 512
        x = self.norm1(x+residual_x) # 30 x 200 x 512
        
        residual_x = x # 30 x 200 x 512
        x = self.fnn(x) # 30 x 200 x 512
        x = self.dropout2(x) # 30 x 200 x 512
        x = self.norm2(x+residual_x) # 30 x 200 x 512 
        return x # this x will be much more context aware that the x we inputed 

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self.x):
        x = self.layers(x)
        return x