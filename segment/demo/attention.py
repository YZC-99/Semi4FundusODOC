import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        mid_channels = int((channels/2)**0.5) # 1 4 9
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        avg = self.sharedMLP(self.avg_pool(x))
        x = x * torch.sigmoid(avg)
        return x

if __name__ == '__main__':

    input = torch.randn(2,1024,32,32)
    att = Attention(1024)
    out,avg = att(input)
    print(out.size())
    print(avg.size())