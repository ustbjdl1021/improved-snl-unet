import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from model.snl_block import ImprovedSNL


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Use tensors of different dimensions, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarization
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
   DropPath (random depth) for each sample (applied to the main path of the residual block).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=(512,512), patch_size=16, n_channels=3, embed_dim=48, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(128), 
                                  nn.ReLU(inplace=True))
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(256), 
                                  nn.ReLU(inplace=True))
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(512), 
                                  nn.ReLU(inplace=True))                                                                              
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.out = nn.Sequential(nn.Conv2d(512, 48, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(48), 
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.bn(x_)
        x1 = self.relu(x_)
        x1_ = self.maxpool1(x1)
        x2 = self.conv_bn_relu1(x1_)
        x2_ = self.maxpool2(x2)
        x3 = self.conv_bn_relu2(x2_)
        x3_ = self.maxpool3(x3)
        x4 = self.conv_bn_relu3(x3_)

        x_encoder_out = self.out(self.maxpool4(x4))

        return x_encoder_out, x1, x2, x3, x4


class LinearConv(nn.Module):
    def __init__(self):
        super(LinearConv, self).__init__()
        self.linear_conv = nn.Linear(48, 48)

    def forward(self, x):
        x1 = x.permute(0, 2, 3, 1)
        # x1 = self.linear_conv(x1)
        x1 = x1.flatten(1, 2)

        return x1


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True))
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outconv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x,  x1, x2, x3, x4):
        y_ = self.conv1(x)
        y_ = self.bn1(y_)
        y_ = self.ReLU1(y_)
        y4 = self.up1(y_)
        y4 = torch.cat((y4, x4), dim=1)
        y4 = self.conv_bn_relu1(y4)
        y3 = self.up2(y4)
        y3 = torch.cat((y3, x3), dim=1)
        y3 = self.conv_bn_relu2(y3)
        y2 = self.up3(y3)
        y2 = torch.cat((y2, x2), dim=1)
        y2 = self.conv_bn_relu3(y2)
        y1 = self.up4(y2)
        y1 = torch.cat((y1, x1), dim=1)
        y_out = self.outconv(y1)

        return y_out


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(688,688), patch_size=16, n_channels=3, n_classes=2,
                 embed_dim=48, depth=2, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, encoder=Encoder, linearconv=LinearConv):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            n_channels (int): number of input channels
            n_classes (int): number of classes for head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_features = self.embed_dim = embed_dim  # num_features 为了与其他模型保持一致
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.encoder = encoder(in_channels=3)
        self.linearconv = linearconv()
        self.need_patch_embed = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, n_channels=n_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        #  head(s)
        # self.head = nn.Linear(self.num_features, n_classes) if n_classes > 0 else nn.Identity()

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.n_classes) if n_classes > 0 else nn.Identity()

        # output conv
        self.outconv = OutConv(in_channels=self.embed_dim, out_channels=self.n_classes)

        self.snl = ImprovedSNL(self.embed_dim, 4)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # if self.dist_token is None:
        #     # [B, C, H, W] -> [B, num_patches, embed_dim]
        #     if self.need_patch_embed:
        #         x = self.patch_embed(x)  # [B, 196, 768]
        #     else:
        x_encoder_out, x1, x2, x3, x4 = self.encoder(x)
        x_out = self.linearconv(x_encoder_out) # [B, 43*43, 48 ]
        # else:
        #     x = torch.cat(( self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x_out = self.pos_drop(x_out + self.pos_embed)
        x_out = self.blocks(x_out)
        x_out = self.norm(x_out)

        return x_out, x1, x2, x3, x4

    def out(self, x, x1, x2, x3, x4):
        x = x.reshape(-1, self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size, self.embed_dim)
        x = x.permute(0, 3, 1, 2).contiguous()

        # use snl block
        # x = self.snl(x)

        x = self.outconv(x, x1, x2, x3, x4)
        return x

    def forward(self, x):
        x, x1, x2, x3, x4 = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.out(x, x1, x2, x3, x4)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == '__main__':
    with torch.no_grad():
        cuda0 = torch.device('cuda:0')
        x = torch.rand((12, 3, 512, 512), device=cuda0)
        model = VisionTransformer(img_size=(512,512), patch_size=16, n_channels=3, n_classes=2,
                 embed_dim=48, depth=4, num_heads=12, mlp_ratio=4.0).cuda(device=cuda0)
        out = model(x)
        print(out.shape)