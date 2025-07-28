import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torch.amp import autocast 
import copy
import os
import wget
from timm.models.layers import to_2tuple, trunc_normal_
from torchvision.models import ResNet18_Weights
from torchvision.models.video import R3D_18_Weights

class LeadPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class LeadAttention_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        pooled = torch.cat([avg_pool, max_pool], dim=1)

        attention = self.conv(pooled)
        attention = self.sigmoid(attention)

        return x * attention

class Pairwise_Attn_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), 
                             padding=(0, kernel_size//2, kernel_size//2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(pooled)
        attention = self.sigmoid(attention)
        
        return x * attention



class LeadAttention3D(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.resnet3d = models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.resnet3d.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), 
                                         stride=(1, 2, 2), padding=(1, 3, 3), 
                                         bias=False)
        self.Pairwise_Attn = Pairwise_Attn_module()
        
        in_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Pairwise_Attn(x)
        return self.resnet3d(x)


class LocalFeatureTransformer(nn.Module):
    def __init__(self, label_dim=768, fstride=10, tstride=10, input_fdim=128, input_tdim=128, 
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):
        super().__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5'
        timm.models.vision_transformer.PatchEmbed = LeadPatchEmbed

        if audioset_pretrain == False:
            if model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be base384.')
            
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), 
                                        nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), 
                                     stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(
                    self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, 
                          self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim,
                          self.oringal_hw, self.oringal_hw)
            
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): 
                                            int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim),
                                                              mode='bilinear')
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): 
                                            int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim),
                                                              mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=128):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16),
                             stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast('cuda')
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x

class LocalFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet2d = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet2d.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet2d.children())[:-2])
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)
        self.resize = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.lead_attn = LeadAttention_module()

    def forward(self, x):
        x = self.features(x)
        x = self.lead_attn(x)
        x = self.final_conv(x)
        x = self.resize(x)
        return x

class LeadOverlappingEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=768, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        num_patches = ((img_size[1] - patch_size[1]) // stride[1] + 1) * \
                     ((img_size[0] - patch_size[0]) // stride[0] + 1)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class GlobalFeatureTransformer(nn.Module):
    def __init__(self, num_classes, img_size=(4, 1), patch_size=(2, 1), stride=(1, 1),
                 in_chans=768, embed_dim=768):
        super().__init__()
        self.patch_embed = LeadOverlappingEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,
                                   num_classes=num_classes)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        self.head = self.vit.head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        embedding = x[:, 0]
        return embedding

class PairwiseLeadAttention3D(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.resnet3d_pair1 = models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.resnet3d_pair2 = models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        
        self.resnet3d_pair1.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 7, 7), 
                                               stride=(1, 2, 2), padding=(1, 3, 3), 
                                               bias=False)
        self.resnet3d_pair2.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 7, 7), 
                                               stride=(1, 2, 2), padding=(1, 3, 3), 
                                               bias=False)
        
        self.Pair_attn_Left = Pairwise_Attn_module()
        self.Pair_attn_Right = Pairwise_Attn_module()

        in_features = self.resnet3d_pair1.fc.in_features
        output_layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
        
        self.resnet3d_pair1.fc = output_layers
        self.resnet3d_pair2.fc = copy.deepcopy(output_layers)

    def forward(self, x):
        pair1 = x[:, :, :2, :, :]
        pair2 = x[:, :, 2:, :, :]

        pair1 = self.Pair_attn_Left(pair1)
        pair2 = self.Pair_attn_Right(pair2)
        
        weights_pair1 = self.resnet3d_pair1(pair1)
        weights_pair2 = self.resnet3d_pair2(pair2)
        
        return weights_pair1, weights_pair2
    
class ESCAViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.resnet3d = models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.resnet3d.stem[0] = nn.Conv3d(2, 64, kernel_size=(3, 7, 7), 
                                         stride=(1, 2, 2), padding=(1, 3, 3), 
                                         bias=False)
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Linear(num_features, 256)

        self.lead_attention = LeadAttention3D(output_dim=4)
        self.pairwise_attention = PairwiseLeadAttention3D(output_dim=2)
        
        self.local_encoder = LocalFeatureEncoder()
        self.local_transformer = LocalFeatureTransformer(
            label_dim=768,
            input_tdim=128,
            input_fdim=128,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size='base384'
        )

        #ast_checkpoint = 'speechcommands_10_10_0.9812.pth'
        ast_checkpoint = r"D:\EEG_After\Data\speechcommands_10_10_0.9812.pth"
        if ast_checkpoint:
            print(f"Loading AST checkpoint from {ast_checkpoint}")
            checkpoint = torch.load(ast_checkpoint, map_location='cpu', weights_only=True)
            self.local_transformer.load_state_dict(checkpoint, strict=False)
            print("AST checkpoint loaded successfully")
        
        self.global_transformer = GlobalFeatureTransformer(
            num_classes=num_classes,
            img_size=(4, 1),
            patch_size=(2, 1),
            stride=(1, 1),
            in_chans=768,
            embed_dim=768
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 378),
            nn.GELU(),
            nn.Linear(378, 189),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(189, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        Left_Channel = x[:, :2, :, :, :]
        Right_Channel = x[:, 2:, :, :, :]
        
        left_x = self.resnet3d(Left_Channel)
        right_x = self.resnet3d(Right_Channel)

        attention_weights = self.lead_attention(x.transpose(1, 2))
        pair1_weights, pair2_weights = self.pairwise_attention(x.transpose(1, 2))

        left_weight = torch.sum(pair1_weights, axis=1, keepdims=True)
        right_weight = torch.sum(pair2_weights, axis=1, keepdims=True)

        left_x = left_x * left_weight
        right_x = right_x * right_weight
        features_list = []
        for i in range(4):
            lead_input = x[:, i]
            spatial_features = self.local_encoder(lead_input)
            spectral_input = spatial_features.squeeze(1).transpose(1, 2)
            lead_features = self.local_transformer(spectral_input)
            features_list.append(lead_features)
        
        stacked_features = torch.stack(features_list, dim=1)

        attention_weights = attention_weights.unsqueeze(-1)
        weighted_features = stacked_features * attention_weights
        
        pair_weights = torch.cat([pair1_weights, pair2_weights], dim=1).unsqueeze(-1)
        weighted_features = weighted_features * pair_weights

        transformer_input = weighted_features.transpose(1, 2)
        transformer_input = transformer_input.unsqueeze(-1)
        embedding = self.global_transformer(transformer_input)
        
        feature = torch.cat((left_x, right_x, embedding), dim=1)
        embedding = self.fc(feature)
        output = self.mlp_head(embedding)
        
        return output, embedding
    
# def test_model(device='cuda'):
#     print(f"Using device: {device}")
#     batch_size = 2
#     x = torch.randn(batch_size, 4, 1, 256, 256).to(device)
#     print(f"Input shape: {x.shape}")
#     model = ESCAViT(num_classes=6).to(device)
#     model.eval()
#     with torch.no_grad():
#         output, embedding = model(x)
#         print(f"Output shape: {output.shape}")
#         print(f"embedding shape: {embedding.shape}")
#         print(f"Output device: {output.device}")
#     return output

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ESCAViT(num_classes=6).to(device)
#     print(f"Number of trainable parameters in the model: {count_parameters(model)}")
#     test_model()


