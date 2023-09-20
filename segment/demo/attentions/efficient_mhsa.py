import torch.nn as nn
import torch
embed_dim = 768
mham = nn.MultiheadAttention(embed_dim=embed_dim // 8, num_heads=4,dropout=0.05, batch_first=True)
query = torch.randn(3,embed_dim,64,64)
key = torch.randn(3,embed_dim,64,64)
value = torch.randn(3,embed_dim,64,64)

b,c,h,w = query.size()


query_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 8, kernel_size=1)
key_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 8, kernel_size=1)
value_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim // 8, kernel_size=1)



down_query = query_conv(query).contiguous().view(b,-1,c // 8)
down_key = key_conv(key).contiguous().view(b,-1,c // 8)
down_value = value_conv(value).contiguous().view(b,-1,c // 8)

out,_ = mham(down_query,down_key,down_value)
print(down_query.shape)
print(out.shape)
