import math
import torch
from utils.box import make_anchors

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))

class DFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)
        
        # Add memory optimization for HD
        self.chunk_size = 1024  # Adjustable based on available memory

    def forward(self, x):
        b, c, a = x.shape
        
        # Memory efficient processing for HD resolution
        if a > self.chunk_size:
            return self.chunked_forward(x, b, c, a)
            
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)
    
    def chunked_forward(self, x, b, c, a):
        # Process in chunks for memory efficiency
        chunk_size = self.chunk_size
        outputs = []
        
        for i in range(0, a, chunk_size):
            end = min(i + chunk_size, a)
            chunk = x[..., i:end]
            chunk = chunk.view(b, 4, self.ch, -1).transpose(2, 1)
            output = self.conv(chunk.softmax(1)).view(b, 4, -1)
            outputs.append(output)
            
        return torch.cat(outputs, dim=2)


class DFLHead(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc, img_size, interchannels, filters, mode='decoupled'):
        super().__init__()
        assert mode in ['coupled', 'decoupled'], "wrong dfl head mode"
        self.mode = mode
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.img_size = img_size  # (1080, 1920) for HD
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.interchannels = interchannels
        
        # Adjusted for HD resolution
        self.grid_size = (img_size[0] // 32, img_size[1] // 32)  # Base grid size for HD
        
        self.dfl = DFL(self.ch)
        
        # Modified convolution layers for HD resolution
        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(
                Conv(x, self.interchannels, 3),
                Conv(self.interchannels, self.interchannels, 3),
                torch.nn.Conv2d(self.interchannels, self.nc, 1)
            ) for x in filters
        )
        
        self.box = torch.nn.ModuleList(
            torch.nn.Sequential(
                Conv(x, self.interchannels, 3),
                Conv(self.interchannels, self.interchannels, 3),
                torch.nn.Conv2d(self.interchannels, 4 * self.ch, 1)
            ) for x in filters
        )

    def forward(self, x):
        if self.training:
            if self.mode == 'coupled':
                for i in range(self.nl):
                    x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
            elif self.mode == 'decoupled':
                for i in range(self.nl):
                    x[i] = torch.cat((self.box[i](x[i][0]), self.cls[i](x[i][1])), 1)
            return x
            
        # Memory efficient inference for HD
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        
        # Process in chunks for memory efficiency
        batch_size = x[0].shape[0]
        outputs = []
        chunk_size = 8192  # Adjustable based on available memory
        
        for i in range(0, self.nl):
            if self.mode == 'coupled':
                feat = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
            else:  # decoupled
                feat = torch.cat((self.box[i](x[i][0]), self.cls[i](x[i][1])), 1)
            outputs.append(feat.view(batch_size, self.no, -1))
        
        x = torch.cat(outputs, 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        
        # Process DFL in chunks
        dfl_out = self.dfl(box)
        a, b = torch.split(dfl_out, 2, 1)
        
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases with HD-appropriate values
        for a, b, s in zip(self.box, self.cls, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            # Adjusted for HD resolution
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (
                (self.img_size[0] * self.img_size[1]) / (s * s)
            ))
