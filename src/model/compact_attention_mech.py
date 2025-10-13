import torch.nn as nn 
from torch import torch



class CompactAttentionMech(nn.Module):


    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        querys = x @ self.W_query
        values = x @ self.W_value
        attn_scores = querys @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values 
        return context_vec

inputs = torch.rand(2, 3)
sa_v1 = CompactAttentionMech(3,3)

print(sa_v1(inputs))