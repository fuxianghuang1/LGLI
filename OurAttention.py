import torch
import torch.nn as nn
import torchvision


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, len_text, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.image_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.text_MLP = nn.Sequential(
            nn.Linear(len_text, channel),
            nn.ReLU(),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        ) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        avgout = self.image_MLP(self.avg_pool(x))
        maxout = self.image_MLP(self.max_pool(x))
        if t is None:
           return self.sigmoid(avgout + maxout)
        else:
           t = self.text_MLP(t).view(maxout.shape)
           return self.sigmoid(t*(avgout + maxout))


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class OurAttention(nn.Module):
    def __init__(self, channel, len_text, alpha = None):
        super(OurAttention, self).__init__()
        self.alpha = alpha
        self.channel_attention = ChannelAttentionModule(channel, len_text)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x, t = None):
        if self.alpha is None:
            out = x
            #print('out:', out, 'x:', x)
        else:    
            out = self.channel_attention(x, t) * x
            out = self.alpha * self.spatial_attention(out) * out + x
            #print('aaa', self.alpha)
        return out
