import math
import torch 
from torch import nn 
from torch import Tensor 
import torch.nn.functional as F 

class GELU(nn.Module):
    def __init__(self, approximate:str = 'none')->None: 
        super().__init__()
        self.approximate = approximate 
    
    def forward(self, input:Tensor)->Tensor:
        return 0.5 * input * (1 + torch.tanh(math.sqrt(math.pi / 2) * (input + 0.044715 * input ** 3)))


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim 
        self.num_heads = config.num_heads 
        assert embed_dim % self.num_heads == 0, "Invalid heads and embedding dimensions"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # In the decoder we need a mask layer, but we don't want that to be trained. So, it is used as self.register buffer
        # See torch.tril for more details
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len)).unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0,2,3,1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1,2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1,2)

        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        mask = self.mask[:,:,:seq_len,:seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_dropout(attn)
        attn = F.softmax(attn, dim = -1)
        y = torch.matmul(attn, v)
        y = y.transpose(1,2)
        y = y.transpose(1, 2)
        y = y.reshape(batch_size, seq_len, -1)
        y = self.proj_dropout(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            GELU(), 
            nn.Linear(embed_dim * 4, embed_dim), 
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x+ self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.max_len = config.max_len
        self.tok_embed = nn.Embedding(
            config.vocab_size, embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, embed_dim)
        )
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, config.vocab_size)

    def forward(self, x, target = None):
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "Sequence longer than model's maximum length "

        tok_embedding = self.tok_embed(x)
        pos_embedding = self.pos_embed[:,:seq_len, :]

        x = self.dropout(tok_embedding + pos_embedding)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x) 
        return x