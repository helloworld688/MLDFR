import timm
import torch
import torch.nn as nn
from .layers import Mlp, DropPath, trunc_normal_
from functools import partial


class TalkingHeadAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScaleBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=TalkingHeadAttn,
            mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_1 * self.mlp(self.norm2(x)))
        return x

class block(nn.Module):
    def __init__(
            self, img_size=384, patch_size=16,
            embed_dim=448, depth=5, num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            block_layers=LayerScaleBlock,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            attn_block=TalkingHeadAttn,
            mlp_block=Mlp,
            init_values=1e-5,
    ):
        super().__init__()
        self.grad_checkpointing = False
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.Sequential(*[
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, attn_block=attn_block, mlp_block=mlp_block, init_values=init_values)
            for i in range(depth)])

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MultiScaleModel(torch.nn.Module):
    def __init__(self,backbone_name,block_k,chanel):
        super(MultiScaleModel, self).__init__()
        self.backbone_name = backbone_name
        self.block_k = block_k
        self.feature_extractor = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=[1, 2, 3])
        self.pool = torch.nn.AvgPool2d(2, stride=2)
        self.model_close = timm.create_model('cait_xs24_384', pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model_close.parameters():
            param.requires_grad = False
        self.featureRec = block(embed_dim=chanel, depth=self.block_k)
        
    def forward(self, x, z, train_flag):
        # cnn_extract
        self.feature_extractor.eval()
        CNN_features_x = self.feature_extractor(x)
        CNN_features_x[0] = self.pool(CNN_features_x[0])
        CNN_features_x[0] = self.pool(CNN_features_x[0])
        CNN_features_x[1] = self.pool(CNN_features_x[1])
        CNN_features_x = torch.cat((CNN_features_x[0], CNN_features_x[1], CNN_features_x[2]), 1)
        N, C, H, W = CNN_features_x.shape
        CNN_features_x = CNN_features_x.reshape(N, C, H * W)

        CNN_features_z = self.feature_extractor(z)
        CNN_features_z[0] = self.pool(CNN_features_z[0])
        CNN_features_z[0] = self.pool(CNN_features_z[0])
        CNN_features_z[1] = self.pool(CNN_features_z[1])
        CNN_features_z = torch.cat((CNN_features_z[0], CNN_features_z[1], CNN_features_z[2]), 1)
        N, C, H, W = CNN_features_z.shape
        CNN_features_z = CNN_features_z.reshape(N, C, H * W)

        #cait_extract
        self.model_close.eval()
        with torch.no_grad():
            x = self.model_close.patch_embed(x)
            x = x + self.model_close.pos_embed
            x = self.model_close.pos_drop(x)
            for i in range(15):
                x = self.model_close.blocks[i](x)
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            CaiT_features_x = x

            z = self.model_close.patch_embed(z)
            x = z + self.model_close.pos_embed
            x = self.model_close.pos_drop(x)
            for i in range(5):
                x = self.model_close.blocks[i](x)
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            CaiT_features_z = x
        # Fusion
        features_x = torch.cat((CNN_features_x, CaiT_features_x), 1)
        features_z = torch.cat((CNN_features_z, CaiT_features_z), 1)
        # Restore
        if train_flag:
            self.featureRec.train()
        else:
            self.featureRec.eval()
        features_z = features_z.permute(0, 2, 1)
        x = features_z + self.featureRec.pos_embed
        x = self.featureRec.pos_drop(x)
        for i in range(self.block_k):
            x = self.featureRec.blocks[i](x)
        N, _, C = x.shape
        x = x.permute(0, 2, 1)
        features_z = x

        return features_x, features_z


