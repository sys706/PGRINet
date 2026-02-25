from torch import nn
import torch
import math
from torch.nn import functional as F
from model.base_function_dface import init_net
from model.base_function import init_net as init_net_base
from einops import rearrange as rearrange
import numbers
from .base_function import *

# encoder
def define_e(init_type='normal', gpu_ids=[]):
    net = Encoder(ngf=48, L=6)
    return init_net(net, init_type, gpu_ids)

# decoder
def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=48)
    return init_net(net, init_type, gpu_ids)

# discriminator
def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', use_spect=True, use_coord=False,
             use_attn=True,  init_type='orthogonal', gpu_ids=[]):

    net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)

    return init_net_base(net, init_type, activation, gpu_ids)


class Encoder(nn.Module):
    def __init__(self, ngf=48, L=6, num_block=[1,2,3,4], num_head=[1,2,4,8], factor=2.66):
        super(Encoder, self).__init__()
        self.L = L

        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=7, padding=0),
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

        self.posterior = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=num_head[3], expansion_factor=1)]
        )

        self.z_nc = ngf * 4

    def forward(self, img_m, img_c=None):
        noise = torch.normal(mean=torch.zeros_like(img_m), std=torch.ones_like(img_m) * (1. / 128.))
        img_m = img_m + noise
        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m
        feature256 = self.start(img)
        feature256 = self.trane256(feature256)
        feature128 = self.down128(feature256)
        feature128 = self.trane128(feature128)
        feature64 = self.down64(feature128)
        feature64 = self.trane64(feature64)
        feature32 = self.down32(feature64)
        feature32 = self.trane32(feature32)

        if type(img_c) != type(None):
            distribution = self.two_paths(feature32)
            return distribution, feature32
        else:
            distribution = self.one_path(feature32)
            return distribution, feature32

    def one_path(self, f_in):
        """one path for baseline training or testing"""
        f_m = f_in
        distribution = []
        f_m = self.infer_prior(f_m)

        o = self.prior(f_m)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution

    def two_paths(self, f_in):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)

        distributions = []

        o = self.posterior(f_c)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)
        distribution = self.one_path(f_m)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        return distributions

class Generator(nn.Module):
    def __init__(self, ngf=48, num_block=[1,2,3,4], num_head=[1,2,4,8], factor=2.66):
        super(Generator, self).__init__()

        self.start = nn.Sequential(
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.GELU(),
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 8, kernel_size=1, stride=1, bias=False)
        )

        self.up64 = Upsample(ngf*8)  # B *4ngf * 64, 64
        self.fuse64 = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*4, kernel_size=1, stride=1, bias=False)
        self.trand64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*4, head=num_head[2],expansion_factor=factor) for i in range(num_block[2])]
        )

        self.up128 = Upsample(ngf*4) # B *2ngf * 128, 128
        self.fuse128 = nn.Conv2d(in_channels=2*ngf, out_channels=2*ngf, kernel_size=1, stride=1, bias=False)
        self.trand128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*2, head=num_head[1],expansion_factor=factor) for i in range(num_block[1])]
        )

        self.up256 = Upsample(ngf*2) # B *ngf * 256, 256
        self.fuse256 = nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=1, stride=1)
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
        z = self.start(z)
        z = z + f

        out64 = self.up64(z)
        out64 = self.fuse64(out64)
        out64 = self.trand64(out64)
        out128 = self.up128(out64)
        out128 = self.fuse128(out128)
        out128 = self.trand128(out128)

        out256 = self.up256(out128)
        out256 = self.fuse256(out256)
        out256 = self.trand256(out256)
        out = torch.tanh(self.out(out256))
        return out

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

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.relu(attn)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out * g
        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66,LayerNorm_type='WithBias'):
        super().__init__()

        num_ch = int(dim * expansion_factor)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.start1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.start2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, stride=1, padding=1,
                      groups=dim * 2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.start1(x)
        x2 = self.start2(x)
        out = x1 * x2
        out = self.norm(out)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
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

class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module