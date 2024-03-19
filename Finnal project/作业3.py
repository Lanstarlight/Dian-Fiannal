import torch
import torch.nn.functional as nn


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        #线性层
        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        #输出层
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, return_attention_weights=False):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        #数据重塑
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        #线性变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)


        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = nn.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)


        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        if return_attention_weights:
            return out, attention
        else:
            return out


# 测试随机数据
embed_size = 256 #嵌入大小
heads = 8 #注意力头

# 创建随机的values、keys和queries张量
values = torch.rand((5, 60, embed_size))
keys = torch.rand((5, 60, embed_size))
queries = torch.rand((5, 40, embed_size))

# 实例化MultiHeadAttention模型
attention_model = MultiHeadAttention(embed_size, heads)

# 调用模型并获取输出和注意力权重
output, attention_weights = attention_model(values, keys, queries, return_attention_weights=True)

print("Output shape:", output.shape)  # Expected: [5, 40, 256]
print("Attention weights shape:", attention_weights.shape)  # Expected: [5, num_heads, 40, 60] or similar based on heads
