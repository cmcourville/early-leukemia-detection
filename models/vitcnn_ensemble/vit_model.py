import torch
from torch import nn

class VitMultiAttentionHead(nn.Module):
    def __init__(self, embedded_dimension: int = 256, number_of_heads: int = 8, dropout_rate: float = 0.2):
        super().__init__()
        self.num_heads = number_of_heads
        self.head_dim = embedded_dimension // number_of_heads
        self.scale = self.head_dim ** -0.5
        # Store the Query,Key,Value in one Matrix for performance benefits
        self.query_key_value = nn.Linear(embedded_dimension, embedded_dimension * 3)
        self.projection = nn.Linear(embedded_dimension, embedded_dimension)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dimension = x.shape
        query_key_value = self.query_key_value(x).reshape(batch, seq, 3, self.num_heads, self.head_dim)
        # Reordering the Matrix such that, the 3 segments are the first Dim
        # Batch is the second dim, Number of heads is the third dim, Sequence is the foruth dim, and head dim is last
        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        output = (attention @ value).transpose(1, 2).reshape(batch, seq, dimension)
        output = self.projection(output)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int = 8, embedding_dimension: int = 256, dropout_rate: float = 0.2,
                 feedforward_dimension: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.multi_attention_head = VitMultiAttentionHead(embedding_dimension, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embedding_dimension, feedforward_dimension),
            nn.GELU(),
            nn.Linear(feedforward_dimension, embedding_dimension),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_attention_head(self.norm1(x))
        x = x + self.feed_forward_network(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self,
                 input_dimension: int = 2048,
                 num_heads: int = 8,
                 embedding_dimension: int = 256,
                 dropout_rate: float = 0.2,
                 feedforward_dimension: int = 256,
                 input_length: int = 25):
        super().__init__()
        # Consuming the ouput of the CNN Backbone into the 256 dimension used for the ViT
        self.input_projection = nn.Linear(input_dimension, embedding_dimension)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimension) * 0.02)
        self.pos_embedded = nn.Parameter(torch.randn(1, input_length + 1, embedding_dimension) * 0.02)
        self.positional_dropout = nn.Dropout(p=dropout_rate)
        # Two transformer layers mentioned in the paper
        self.blocks = nn.Sequential(
            TransformerBlock(num_heads, embedding_dimension, dropout_rate, feedforward_dimension),
            TransformerBlock(num_heads, embedding_dimension, dropout_rate, feedforward_dimension)
        )
        self.normalize = nn.LayerNorm(embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.input_projection(x)
        cls_token = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedded[:, : x.shape[1], :]
        x = self.positional_dropout(x)
        x = self.blocks(x)
        x = self.normalize(x)
        return x[:, 0]
