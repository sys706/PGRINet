from torch import nn
import torch
import math
# from model.swish import Swish
from torch.nn import functional as F
from model.base_function_dface import init_net
from einops import rearrange as rearrange
import numbers

# encoder
def define_e(init_type='normal', gpu_ids=[]):
    net = Encoder(ngf=48, L=6)
    return init_net(net, init_type, gpu_ids)

# decoder
def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=48)
    return init_net(net, init_type, gpu_ids)

# discriminator
def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)


class Encoder(nn.Module):
    def __init__(self, ngf=48, L=6, num_block=[1,2,3,4], num_head=[1,2,4,8], factor=2.66):
        super(Encoder, self).__init__()
        self.L = L

        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=6, out_channels=ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )
        self.trane256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0], expansion_factor=factor) for i in range(num_block[0])]
        )
        self.down128 = Downsample(num_ch=ngf)  # B *2ngf * 128, 128
        self.trane128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 2, head=num_head[1], expansion_factor=factor) for i in range(num_block[1])]
        )
        self.down64 = Downsample(num_ch=ngf * 2)  # B *4ngf * 64, 64
        self.trane64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 4, head=num_head[2], expansion_factor=factor) for i in range(num_block[2])]
        )
        self.down32 = Downsample(num_ch=ngf * 4)  # B *8ngf * 32, 32
        self.trane32 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=num_head[3], expansion_factor=factor) for i in range(num_block[3])]
        )

        # inference part
        self.infer_prior = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=num_head[3], expansion_factor=1) for i in range(L)]
        )
        self.prior = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=num_head[3], expansion_factor=1)]
        )

        self.z_nc = ngf * 4

    def forward(self, x, mask=None):
        noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 128.))
        x = x + noise
        feature = torch.cat([x, mask], dim=1)
        # print("the shape of feature: ", feature.size()) # [2, 6, 256, 256]
        feature256 = self.start(feature)
        feature256 = self.trane256(feature256)
        feature128 = self.down128(feature256)
        feature128 = self.trane128(feature128)
        feature64 = self.down64(feature128)
        feature64 = self.trane64(feature64)
        feature32 = self.down32(feature64)
        feature32 = self.trane32(feature32)
        # print("the shape of feature32: ", feature32.size()) # [2, 48*8, 32, 32]

        distribution = []
        f_m = self.infer_prior(feature32)
        # print("the shape of f_m: ", f_m.size()) # [2, 48*8, 32, 32]
        o = self.prior(f_m)
        # print("the shape of o: ", o.size()) # [2, 48*8, 32, 32]
        mu, std = torch.split(o, self.z_nc, dim=1)
        # print("the shape of mu: ", mu.size()) # [2, 48*4, 32, 32]
        # print("the shape of std: ", std.size()) # [2, 48*4, 32, 32]
        distribution.append([mu, F.softplus(std)])

        return distribution, [feature32,feature64,feature128,feature256]

class Generator(nn.Module):
    def __init__(self, ngf=48, num_block=[1,2,3,4], num_head=[1,2,4,8], factor=2.66):
        super(Generator, self).__init__()

        # self.start = nn.Sequential(
        #     nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=1, stride=1, bias=False),
        #     nn.InstanceNorm2d(ngf*8),
        #     nn.GELU()
        # )

        # Two
        self.start = nn.Sequential(
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 8, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.GELU(),
            nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=1, stride=1, bias=False),
        )

        self.up64 = Upsample(ngf*8)  # B *4ngf * 64, 64
        self.fuse64 = nn.Conv2d(in_channels=ngf*4*2, out_channels=ngf*4, kernel_size=1, stride=1, bias=False)
        self.trand64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*4, head=num_head[2],expansion_factor=factor) for i in range(num_block[2])]
        )

        self.up128 = Upsample(ngf*4) # B *2ngf * 128, 128
        self.fuse128 = nn.Conv2d(in_channels=4*ngf, out_channels=2*ngf, kernel_size=1, stride=1, bias=False)
        self.trand128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*2, head=num_head[1],expansion_factor=factor) for i in range(num_block[1])]
        )

        self.up256 = Upsample(ngf*2) # B *ngf * 256, 256
        self.fuse256 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, stride=1)
        self.trand256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
        )

        self.trand2562 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
        )

    def forward(self, z, f=None):
        # print("the shape of z: ", z.size()) # [2, 48*4, 32, 32]
        feature32, feature64, feature128, feature256 = f
        # print("the shape of feature32: ", feature32.size()) # [2, 48*8, 32, 32]
        # print("the shape of feature64: ", feature64.size()) # [2, 48*4, 64, 64]
        # print("the shape of feature128: ", feature128.size()) # [2, 48*2, 128, 128]
        # print("the shape of feature256: ", feature256.size()) # [2, 48, 256, 256]

        z = self.start(z)
        z = z + feature32

        out64 = self.up64(z)
        out64 = self.fuse64(torch.cat([feature64, out64], dim=1))
        out64 = self.trand64(out64)
        #out128 = torch.nn.functional.interpolate(out64, scale_factor=2, mode='nearest')
        out128 = self.up128(out64)
        out128 = self.fuse128(torch.cat([feature128, out128], dim=1))
        out128 = self.trand128(out128)

        out256 = self.up256(out128)
        out256 = self.fuse256(torch.cat([feature256, out256], dim=1))
        out256 = self.trand256(out256)
        #out256 = self.trand2562(out256)
        out = torch.tanh(self.out(out256))
        # print("the shape of out: ", out.size()) # [2, 3, 256, 256]
        return out

# (H * W) * C -> (H/2 * C/2) * (4C) -> (H/4 * W/4) * 16C -> (H/8 * W/8) * 64C
class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=256, head=4, expansion_factor=2.66):
        super().__init__()

        self.attn = Attention_C_M(dim=in_ch, num_heads=head,bias=False,LayerNorm_type='WithBias')
        self.feed_forward = FeedForward(dim=in_ch, expansion_factor=expansion_factor,LayerNorm_type='WithBias')

    def forward(self, x):
        x = self.attn(x) + x
        x = self.feed_forward(x) + x
        return x

class Downsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch*2, track_running_stats=False),
            nn.GELU()
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, num_ch=32):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch//2, track_running_stats=False),
            nn.GELU()
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.body(x)

class Attention_C_M(nn.Module):
    def __init__(self, dim, num_heads, bias,LayerNorm_type):
        super(Attention_C_M, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_1 = self.norm1(x)
        g = self.gate(x_1)

        qkv = self.qkv_dwconv(self.qkv(x_1))
        q, k, v = qkv.chunk(3, dim=1)
        # print("the shape of q: ", q.size()) # [2, 48, 256, 256]
        # print("the shape of k: ", k.size()) # [2, 48, 256, 256]
        # print("the shape of v: ", v.size()) # [2, 48, 256, 256]

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # print("the shape of q1: ", q.size()) # [2, 1, 48, 65536]
        # print("the shape of k1: ", k.size()) # [2, 1, 48, 65536]
        # print("the shape of v1: ", v.size()) # [2, 1, 48, 65536]

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # print("the shape of q2: ", q.size()) # [2, 1, 48, 65536]
        # print("the shape of k2: ", k.size()) # [2, 1, 48, 65536]

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #attn = attn.softmax(dim=-1)
        attn = F.relu(attn)
        #attn = torch.square(attn)
        # print("the shape of temperature: ", self.temperature.size())
        # print("the shape of attn: ", attn.size()) # [2, 1, 48, 48]

        out = (attn @ v)
        # print("the shape of out1: ", out.size()) # [2, 1, 48, 65536]

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # print("the shape of out2: ", out.size()) # [2, 48, 256, 256]

        out = out * g
        out = self.project_out(out)
        # print("the shape of out3: ", out.size()) # [2, 48, 256, 256]
        #out = x+ out
        return out

class FeedForward(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66,LayerNorm_type='WithBias'):
        super().__init__()

        num_ch = int(dim * expansion_factor)
        #self.norm = nn.InstanceNorm2d(num_features=dim, track_running_stats=False)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=num_ch*2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=num_ch*2, out_channels=num_ch*2, kernel_size=3, stride=1, padding=1, groups=num_ch*2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=num_ch, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
        #out = out + x
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module