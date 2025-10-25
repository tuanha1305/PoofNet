import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== LOSS FUNCTIONS ====================
bce_loss = nn.BCELoss(reduction='mean')


def muti_loss_fusion(preds, target):
    """Multi-scale loss fusion"""
    loss0 = 0.0
    loss = 0.0

    for i in range(len(preds)):
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss

    return loss0, loss


# ==================== STYLE ENCODER ====================
class StyleEncoder(nn.Module):
    """
    Encode image style (human/art/graphic design) thành feature vectors
    """

    def __init__(self, num_styles=3, embed_dim=256):
        super().__init__()

        self.num_styles = num_styles
        self.embed_dim = embed_dim

        # Style embedding
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        # MLP để refine style features
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, style_idx):
        """
        Args:
            style_idx: (B,) style indices [0=human, 1=art, 2=graphic_design]
        Returns:
            style_features: (B, embed_dim)
        """
        # Embed style
        style_feat = self.style_embed(style_idx)  # (B, embed_dim)

        # Refine through MLP
        style_feat = self.mlp(style_feat)

        # Normalize
        style_feat = self.norm(style_feat)

        return style_feat


# ==================== BBOX ENCODING MODULE ====================
class BBoxEncoder(nn.Module):
    """
    Encode bounding boxes + class tags thành feature vectors
    """

    def __init__(self, embed_dim=256, num_classes=80):
        super().__init__()

        self.embed_dim = embed_dim

        # Spatial encoding: [x1, y1, x2, y2] -> features
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim // 2)
        )

        # Class/Tag embedding
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim // 2)  # +1 for padding

        # Position encoding for multiple objects
        self.pos_embed = nn.Parameter(torch.randn(1, 20, embed_dim) * 0.02)  # Max 20 objects

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, bboxes, classes):
        """
        Args:
            bboxes: (B, N, 4) normalized [x1, y1, x2, y2] in range [0, 1]
            classes: (B, N) class indices
        Returns:
            bbox_features: (B, N, embed_dim)
        """
        B, N, _ = bboxes.shape

        # Encode spatial information
        spatial_feat = self.spatial_encoder(bboxes)  # (B, N, embed_dim//2)

        # Encode class information
        class_feat = self.class_embed(classes)  # (B, N, embed_dim//2)

        # Concatenate
        bbox_feat = torch.cat([spatial_feat, class_feat], dim=-1)  # (B, N, embed_dim)

        # Add positional encoding
        bbox_feat = bbox_feat + self.pos_embed[:, :N, :]

        # Normalize
        bbox_feat = self.norm(bbox_feat)

        return bbox_feat


# ==================== SPATIAL ATTENTION WITH BBOX ====================
class BBoxSpatialAttention(nn.Module):
    """
    Create spatial attention maps từ bounding boxes
    """

    def __init__(self):
        super().__init__()

    def forward(self, feature_map, bboxes):
        """
        Args:
            feature_map: (B, C, H, W)
            bboxes: (B, N, 4) normalized [x1, y1, x2, y2]
        Returns:
            attention_map: (B, 1, H, W)
        """
        B, C, H, W = feature_map.shape
        _, N, _ = bboxes.shape

        # Create gaussian attention map for each bbox
        attention_map = torch.zeros(B, 1, H, W, device=feature_map.device)

        for b in range(B):
            for n in range(N):
                x1, y1, x2, y2 = bboxes[b, n]

                # Skip padding bboxes (all zeros)
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue

                # Convert normalized coords to pixel coords
                x1_px = int(x1 * W)
                y1_px = int(y1 * H)
                x2_px = int(x2 * W)
                y2_px = int(y2 * H)

                # Clamp to valid range
                x1_px = max(0, min(W - 1, x1_px))
                y1_px = max(0, min(H - 1, y1_px))
                x2_px = max(0, min(W - 1, x2_px))
                y2_px = max(0, min(H - 1, y2_px))

                # Create soft attention (gaussian)
                cx = (x1_px + x2_px) / 2
                cy = (y1_px + y2_px) / 2
                sigma_x = (x2_px - x1_px) / 4 + 1e-6
                sigma_y = (y2_px - y1_px) / 4 + 1e-6

                y_coords = torch.arange(H, device=feature_map.device).float().view(-1, 1)
                x_coords = torch.arange(W, device=feature_map.device).float().view(1, -1)

                gauss = torch.exp(-((x_coords - cx) ** 2 / (2 * sigma_x ** 2) +
                                    (y_coords - cy) ** 2 / (2 * sigma_y ** 2)))

                attention_map[b, 0] = torch.maximum(attention_map[b, 0], gauss)

        return attention_map


# ==================== TRANSFORMER BLOCK WITH BBOX ATTENTION ====================
class TransformerBlockWithBBox(nn.Module):
    """
    Transformer block với cross-attention to bbox features
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, bbox_features):
        """
        Args:
            x: (B, L, C) feature tokens
            bbox_features: (B, N, C) bbox features
        """
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Cross-attention với bbox features
        x = x + self.cross_attn(self.norm2(x), bbox_features, bbox_features)[0]

        # FFN
        x = x + self.mlp(self.norm3(x))

        return x


# ==================== STYLE CONDITIONAL LAYER ====================
class StyleConditionalLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) với style
    Modulate features dựa trên style của ảnh
    """

    def __init__(self, num_channels, style_dim=256):
        super().__init__()

        # Generate scale and shift parameters từ style
        self.scale_shift = nn.Sequential(
            nn.Linear(style_dim, num_channels * 2),
            nn.GELU()
        )

    def forward(self, x, style_features):
        """
        Args:
            x: (B, C, H, W) feature map
            style_features: (B, style_dim) style embedding
        Returns:
            modulated features: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Generate scale and shift
        scale_shift = self.scale_shift(style_features)  # (B, C*2)
        scale, shift = scale_shift.chunk(2, dim=1)  # Each (B, C)

        # Reshape for broadcasting
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)

        # Apply FiLM
        return x * (1 + scale) + shift


# ==================== BASIC CONV BLOCKS (from original ISNet) ====================
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


# ==================== RSU BLOCKS ====================
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


# ==================== MAIN MODEL: ISNet-Transformer với BBox và Style ====================
class ISNetTransformerBBox(nn.Module):
    """
    ISNet-Transformer with BBox, Tag, and Style conditioning

    Architecture:
        - Style Encoder: Encode image style (human/art/graphic)
        - BBox Encoder: Encode bounding boxes and tags
        - CNN Encoder (Stage 1-3): Local feature extraction
        - Transformer Encoder (Stage 4-6): Global context with BBox attention
        - CNN Decoder: Multi-scale fusion with Style and BBox guidance
    """

    def __init__(self, in_ch=3, out_ch=1, num_classes=80, num_styles=3):
        super(ISNetTransformerBBox, self).__init__()

        # ========== Style Encoder ==========
        self.style_encoder = StyleEncoder(num_styles=num_styles, embed_dim=256)

        # ========== BBox Encoder ==========
        self.bbox_encoder = BBoxEncoder(embed_dim=256, num_classes=num_classes)
        self.bbox_spatial_attn = BBoxSpatialAttention()

        # ========== CNN Encoder (Stages 1-3) ==========
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.style_cond1 = StyleConditionalLayer(64, style_dim=256)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.style_cond2 = StyleConditionalLayer(128, style_dim=256)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.style_cond3 = StyleConditionalLayer(256, style_dim=256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # ========== Transformer Encoder (Stages 4-6) ==========
        # Patch embedding for transformer
        self.stage4_conv = nn.Conv2d(256, 512, 3, padding=1)
        self.transformer4 = TransformerBlockWithBBox(512, num_heads=8)
        self.style_cond4 = StyleConditionalLayer(512, style_dim=256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.transformer5 = TransformerBlockWithBBox(512, num_heads=8)
        self.style_cond5 = StyleConditionalLayer(512, style_dim=256)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.transformer6 = TransformerBlockWithBBox(512, num_heads=8)
        self.style_cond6 = StyleConditionalLayer(512, style_dim=256)

        # Project bbox features to match transformer dim
        self.bbox_proj = nn.Linear(256, 512)

        # ========== CNN Decoder ==========
        self.stage5d = RSU4F(1024, 256, 512)
        self.style_cond5d = StyleConditionalLayer(512, style_dim=256)

        self.stage4d = RSU4(1024, 128, 256)
        self.style_cond4d = StyleConditionalLayer(256, style_dim=256)

        self.stage3d = RSU5(512, 64, 128)
        self.style_cond3d = StyleConditionalLayer(128, style_dim=256)

        self.stage2d = RSU6(256, 32, 64)
        self.style_cond2d = StyleConditionalLayer(64, style_dim=256)

        self.stage1d = RSU7(128, 16, 64)
        self.style_cond1d = StyleConditionalLayer(64, style_dim=256)

        # ========== Side Outputs ==========
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # Final fusion
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x, bboxes, classes, style_idx):
        """
        Args:
            x: (B, 3, H, W) input image
            bboxes: (B, N, 4) normalized bounding boxes [x1, y1, x2, y2]
            classes: (B, N) class indices
            style_idx: (B,) style indices [0=human, 1=art, 2=graphic_design]
        Returns:
            list of predicted masks at different scales
        """
        B, _, H, W = x.shape

        # ========== Encode Style ==========
        style_features = self.style_encoder(style_idx)  # (B, 256)

        # ========== Encode BBoxes ==========
        bbox_features = self.bbox_encoder(bboxes, classes)  # (B, N, 256)
        bbox_features_512 = self.bbox_proj(bbox_features)  # (B, N, 512)

        # ========== CNN Encoder with Style Modulation ==========
        hxin = self.conv_in(x)
        # hx = self.pool_in(hxin)

        # Stage 1
        hx1 = self.stage1(hxin)
        hx1 = self.style_cond1(hx1, style_features)  # Apply style modulation
        hx = self.pool12(hx1)

        # Stage 2
        hx2 = self.stage2(hx)
        hx2 = self.style_cond2(hx2, style_features)
        hx = self.pool23(hx2)

        # Stage 3
        hx3 = self.stage3(hx)
        hx3 = self.style_cond3(hx3, style_features)
        hx = self.pool34(hx3)

        # ========== Transformer Encoder with BBox Attention and Style ==========
        # Stage 4
        hx4 = self.stage4_conv(hx)
        hx4 = self.style_cond4(hx4, style_features)
        B, C, H4, W4 = hx4.shape
        hx4_tokens = hx4.flatten(2).transpose(1, 2)  # (B, H4*W4, C)
        hx4_tokens = self.transformer4(hx4_tokens, bbox_features_512)
        hx4 = hx4_tokens.transpose(1, 2).reshape(B, C, H4, W4)
        hx = self.pool45(hx4)

        # Stage 5
        hx5 = self.stage5_conv(hx)
        hx5 = self.style_cond5(hx5, style_features)
        B, C, H5, W5 = hx5.shape
        hx5_tokens = hx5.flatten(2).transpose(1, 2)  # (B, H5*W5, C)
        hx5_tokens = self.transformer5(hx5_tokens, bbox_features_512)
        hx5 = hx5_tokens.transpose(1, 2).reshape(B, C, H5, W5)
        hx = self.pool56(hx5)

        # Stage 6
        hx6 = self.stage6_conv(hx)
        hx6 = self.style_cond6(hx6, style_features)
        B, C, H6, W6 = hx6.shape
        hx6_tokens = hx6.flatten(2).transpose(1, 2)  # (B, H6*W6, C)
        hx6_tokens = self.transformer6(hx6_tokens, bbox_features_512)
        hx6 = hx6_tokens.transpose(1, 2).reshape(B, C, H6, W6)

        # Apply spatial attention from bboxes
        bbox_attn_map = self.bbox_spatial_attn(hx6, bboxes)
        bbox_attn_map = F.interpolate(bbox_attn_map, size=hx6.shape[2:], mode='bilinear', align_corners=True)
        hx6 = hx6 * (1 + bbox_attn_map)  # Amplify regions in bboxes

        # ========== Decoder with Style Modulation ==========
        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5d = self.style_cond5d(hx5d, style_features)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4d = self.style_cond4d(hx4d, style_features)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3d = self.style_cond3d(hx3d, style_features)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2d = self.style_cond2d(hx2d, style_features)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.style_cond1d(hx1d, style_features)

        # ========== Side Outputs ==========
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        # Final fusion
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        # Apply sigmoid
        outputs = [F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2),
                   F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)]

        return outputs

    def compute_loss(self, preds, targets):
        """
        Compute multi-scale loss
        Args:
            preds: list of predictions at different scales
            targets: (B, 1, H, W) ground truth mask
        """
        return muti_loss_fusion(preds, targets)


# ==================== TESTING ====================
if __name__ == "__main__":
    # Test model
    model = ISNetTransformerBBox(in_ch=3, out_ch=1, num_classes=80, num_styles=3)

    # Dummy input
    batch_size = 2
    img = torch.randn(batch_size, 3, 512, 512)
    bboxes = torch.tensor([
        [[0.2, 0.3, 0.5, 0.7], [0.6, 0.2, 0.9, 0.6], [0.0, 0.0, 0.0, 0.0]],  # 2 objects + 1 padding
        [[0.1, 0.1, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]  # 1 object + 2 padding
    ], dtype=torch.float32)
    classes = torch.tensor([
        [0, 2, 0],  # person, car, padding
        [1, 0, 0]  # bicycle, padding, padding
    ], dtype=torch.long)
    style_idx = torch.tensor([0, 1], dtype=torch.long)  # human, art

    # Forward pass
    outputs = model(img, bboxes, classes, style_idx)

    print("Model test successful!")
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output shape: {outputs[0].shape}")

    # Test loss
    gt_mask = torch.randint(0, 2, (batch_size, 1, 512, 512)).float()
    loss0, loss = model.compute_loss(outputs, gt_mask)
    print(f"Loss0: {loss0.item():.4f}, Total loss: {loss.item():.4f}")

    # Test style encoder
    print(f"\nStyle test:")
    style_features = model.style_encoder(style_idx)
    print(f"Style features shape: {style_features.shape}")  # Should be (2, 256)