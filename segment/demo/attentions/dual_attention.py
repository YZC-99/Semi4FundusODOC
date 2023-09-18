from fightingcv_attention.attention.DANet import DAModule
import torch

input=torch.randn(3,768,64,64)
emsa = DAModule(d_model=768,kernel_size=3,H=64,W=64)
output=emsa(input)
print(output.shape)