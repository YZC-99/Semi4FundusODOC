from fightingcv_attention.attention.EMSA import EMSA
import torch

input=torch.randn(3,64*64,768)
emsa = EMSA(d_model=768, d_k=768, d_v=768, h=64,H=64,W=64,ratio=2,apply_transform=True)
output=emsa(input,input,input)
print(output.shape)