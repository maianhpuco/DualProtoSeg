from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.conch_adapter import ConchAdapter


class MultiscaleFeaturePyramid(nn.Module):
    def __init__(self, in_channels=512, target_sizes=[(28, 28), (56, 56), (112, 112), (224, 224)], 
                 target_channels=[256, 128, 64, 32]):
        super().__init__()
        self.target_sizes = target_sizes
        self.target_channels = target_channels
        
        # Projection layers for each pyramid level
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, ch, kernel_size=1, bias=False),
                nn.GroupNorm(16, ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, ch),
                nn.SiLU(inplace=True),
            )
            for ch in target_channels
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps [F1, F2, F3, F4]
                     Each can be [B, C, H, W] at different resolutions
        Returns:
            pyramid: List of 4 feature maps [P1, P2, P3, P4]
                    P1: [B, 256, 28, 28]
                    P2: [B, 128, 56, 56]
                    P3: [B, 64, 112, 112]
                    P4: [B, 32, 224, 224]
        """
        pyramid = []
        for i, (target_size, proj_layer) in enumerate(zip(self.target_sizes, self.projections)):
            # Use feature from corresponding level (F1->P1, F2->P2, etc.)
            # Or use the deepest feature (F4) for all levels if needed
            feat = features[min(i, len(features) - 1)]  # Use F4 for all if i >= len(features)
            
            # Project to target channel dimension
            feat = proj_layer(feat)  # [B, target_ch, H_orig, W_orig]
            
            # Upsample/downsample to target resolution
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            pyramid.append(feat)
        
        return pyramid


class FeatureRefiner(nn.Module):
    def __init__(self, in_channels=768, out_channels=512, depth=8, base_ch=256, dropout=0.1):
        super().__init__()
        self.depth = depth
        
        # 1. Initial projection 768 → base_ch
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=1, bias=False),
            nn.GroupNorm(16, base_ch),
            nn.SiLU(inplace=True)
        )
        
        # 2. Shared residual blocks (depth=8 is the 2025 sweet spot)
        blocks = []
        for i in range(depth):
            blocks.append(nn.Sequential(
                nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, base_ch),
                nn.SiLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, base_ch),
            ))
        self.blocks = nn.ModuleList(blocks)
        
        # 1×1 skip connections (one per block)
        self.skips = nn.ModuleList([
            nn.Conv2d(base_ch, base_ch, kernel_size=1) for _ in range(depth)
        ])
        
        # 3. Final projection to out_channels (512)
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_channels),
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [B, 768, H, W]
        x = self.proj(x)                    # → [B, 256, H, W]
        for block, skip in zip(self.blocks, self.skips):
            identity = skip(x)
            x = block(x)
            x = x + identity                # residual
            x = F.silu(x)
        x = self.final(x)                   # → [B, 512, H, W]
        x = F.normalize(x, dim=1)           # CRITICAL for cosine similarity
        return x


class ClsNetwork(nn.Module):
    def __init__(
        self,
        backbone='mit_b1',
        cls_num_classes=4,
        clip_adapter: Optional[ConchAdapter] = None,
        stride=[4, 2, 2, 1],
        pretrained=True,
        enable_text_fusion=False,
        text_prompts=None,
        fusion_dim=None,
        spatial_agg_all_scales: bool = False,
    ): 
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.clip_adapter = clip_adapter
        self.stride = stride
        self.cam_fusion_levels = (1, 2, 3)
        # Whether to apply spatial aggregation to all CAM scales (cam1‑4) or only cam4.
        # main7 uses the default (False, only cam4); main8 can enable True via config.
        self.spatial_agg_all_scales = spatial_agg_all_scales

        if self.clip_adapter is None:
            raise ValueError("clip_adapter is required")

        # TEXT PROMPTS = PROTOTYPES
        with torch.no_grad():
            all_text_feats, class_text_feats = self.clip_adapter.encode_text(normalize=True)

        self.num_prototypes_per_class = [
            len(prompts) for prompts in self.clip_adapter.prompt_learner.prompts_per_class
        ]   # [3, 3, 3, 3]

        self.total_prototypes = all_text_feats.shape[0]  # 12
        self.prototype_feature_dim = all_text_feats.shape[1]  # 768 from CONCH
        print("> prototype_feature_dim", self.prototype_feature_dim)

        # Text prototypes: (12, 768) - 3 per class from text prompts
        self.register_buffer("prototypes", all_text_feats)           # (12, 768)
        self.register_buffer("class_prototypes", class_text_feats)   # (4, 768)
        
        # Project text prototypes to 512 dim to match learnable image prototypes
        self.text_prototype_dim = 512
        self.text_proj = nn.Linear(self.prototype_feature_dim, self.text_prototype_dim)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        
        # Learnable image prototypes: (12, 512) - 3 per class, learnable
        # Initialize with small random values
        self.learnable_image_prototypes = nn.Parameter(
            torch.randn(self.total_prototypes, self.text_prototype_dim) * 0.02,
            requires_grad=True
        )
        
        # Track if learnable prototypes should be frozen (for warmup)
        self.learnable_prototypes_frozen = False
        
        # Total prototypes per class: 3 (text) + 3 (learnable image) = 6
        # Total prototypes: 12 (text) + 12 (learnable) = 24
        self.total_prototypes_combined = self.total_prototypes * 2  # 24
        
        # Update k_list to reflect 6 prototypes per class
        self.k_list = torch.tensor([num * 2 for num in self.num_prototypes_per_class])  # [6, 6, 6, 6]
        
        self.global_img_proj = nn.Linear(self.text_prototype_dim, self.text_prototype_dim, bias=False)
        # Initialize as identity matrix so it starts as identity mapping
        nn.init.eye_(self.global_img_proj.weight)
        
        # Learnable temperature for contrastive loss (typical range: 0.07-0.2)
        # Increased to 0.1 for better stability
        self.contrastive_temperature = nn.Parameter(torch.tensor(0.1))
        
        # Number of prototypes per class (6: 3 text + 3 learnable)
        self.proto_per_class = self.k_list[0].item()  # 6      

        # Backbone
        trunk = getattr(getattr(self.clip_adapter.model, "visual", None), "trunk", None)
        visual_width = getattr(trunk, "embed_dim", self.prototype_feature_dim)
        self.in_channels = [visual_width] * 4
       
                
        # Logit scales
        self.logit_scale1 = nn.Parameter(torch.ones([]) * (1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * (1 / 0.07))
        self.logit_scale3 = nn.Parameter(torch.ones([]) * (1 / 0.07))
        self.logit_scale4 = nn.Parameter(torch.ones([]) * (1 / 0.07))

        # The features from visual_intermediates are already 512-dim via visual_proj in conch_adapter
        # So we can use them directly without additional projection
        
        # Shared lightweight FPN decoder to refine raw CONCH features (768-dim → 512-dim)
        # Applied to all 4 intermediate layers (F1-F4 → R1-R4)
        self.shared_decoder = FeatureRefiner(
            in_channels=768,      # Raw CONCH feature dimension
            out_channels=512,     # Target dimension (matches prototype dim)
            depth=8,              # Number of residual blocks
            base_ch=256,          # Internal channel dimension
            dropout=0.1
        )
        
        # Creates pyramid with resolutions: 28x28, 56x56, 112x112, 224x224
        # and channels: 256, 128, 64, 32
        self.feature_pyramid = MultiscaleFeaturePyramid(
            in_channels=512,  # Input from refined features R1-R4
            target_sizes=[(28, 28), (56, 56), (112, 112), (224, 224)],
            target_channels=[256, 128, 64, 32]
        )
        
        # Prototype projection layers for each pyramid level
        # Project prototypes from 512-dim to match each pyramid level's channel dimension
        self.proto_projections = nn.ModuleList([
            nn.Linear(self.text_prototype_dim, ch, bias=False)  # 512 -> 256, 128, 64, 32
            for ch in [256, 128, 64, 32]
        ])
        # Initialize prototype projections
        for proj in self.proto_projections:
            nn.init.xavier_uniform_(proj.weight)
       
        self.text_fusion = None 
           

    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    
    def freeze_learnable_prototypes(self):
        """Freeze learnable image prototypes (for warmup phase)"""
        self.learnable_image_prototypes.requires_grad_(False)
        self.learnable_prototypes_frozen = True
        
    def unfreeze_learnable_prototypes(self):
        """Unfreeze learnable image prototypes (after warmup phase)"""
        self.learnable_image_prototypes.requires_grad_(True)
        self.learnable_prototypes_frozen = False
    
    def compute_contrastive_loss(self, global_img_feat, learnable_prototypes, labels):
        """
        Compute Global Contrastive Regularization loss with proper multi-label handling.
        ONLY uses learnable image prototypes (not text prototypes).
        
        Args:
            global_img_feat: [B, 512] - Global image embedding from CONCH
            learnable_prototypes: [12, 512] - Only learnable image prototypes (3 per class)
            labels: [B, num_classes] - Multi-label binary tensor (e.g., [B, 4])
        
        Returns:
            contrastive_loss: scalar tensor
        """
        B = global_img_feat.shape[0]
        
        # Project global image features (initialized to identity)
        projected_img_feat = self.global_img_proj(global_img_feat)  # [B, 512]
        
        # Normalize both image features and prototypes
        projected_img_feat = F.normalize(projected_img_feat, dim=-1)  # [B, 512]
        learnable_prototypes_norm = F.normalize(learnable_prototypes, dim=-1)  # [12, 512]
        
        # Compute cosine similarity: [B, 12] (only learnable prototypes)
        sim = (projected_img_feat @ learnable_prototypes_norm.t()) / self.contrastive_temperature
        
        # Properly handle multi-label classification
        # Create target matrix: [B, 12] where 1 indicates positive prototype, 0 indicates negative
        # For each sample, mark all learnable prototypes of all positive classes as positive
        # Each class has 3 learnable prototypes (not 6 total)
        num_learnable_prototypes_per_class = 3  # Only learnable, not text
        targets = torch.zeros(B, self.total_prototypes, device=labels.device)  # [B, 12]
        
        for class_idx in range(self.cls_num_classes):
            # Get samples that have this class as positive
            class_mask = labels[:, class_idx] > 0.5  # [B] - boolean mask
            
            if class_mask.any():
                # Learnable prototype indices for this class: [class_idx * 3, ..., (class_idx+1) * 3 - 1]
                start_idx = int(class_idx * num_learnable_prototypes_per_class)
                end_idx = int((class_idx + 1) * num_learnable_prototypes_per_class)
                
                # Mark these learnable prototypes as positive for samples with this class
                targets[class_mask, start_idx:end_idx] = 1.0
        
        # Use binary cross-entropy with logits for multi-label loss
        # This properly handles cases where multiple classes (and thus multiple prototype groups) are positive
        contrastive_loss = F.binary_cross_entropy_with_logits(sim, targets, reduction='mean')
        
        return contrastive_loss
    
    def forward(self, x, labels=None):
        # This ensures gradients flow through the learnable prompt context vectors
        with torch.set_grad_enabled(self.training):  # Enable grad if training
            all_text_feats, class_text_feats = self.clip_adapter.encode_text(normalize=True)
        
        # Project text prototypes from 768 to 512 dim
        text_prototypes_proj = self.text_proj(all_text_feats)  # (12, 512)
        text_prototypes_proj = F.normalize(text_prototypes_proj, dim=-1)
        
        # Get learnable image prototypes (12, 512)
        learnable_prototypes = F.normalize(self.learnable_image_prototypes, dim=-1)
        
        # Concatenate text and learnable prototypes: (24, 512)
        # Order: [text_proto_0, learnable_proto_0, text_proto_1, learnable_proto_1, ...]
        # This gives 6 prototypes per class (3 text + 3 learnable)
        combined_prototypes = torch.zeros(
            self.total_prototypes_combined, 
            self.text_prototype_dim,
            device=text_prototypes_proj.device,
            dtype=text_prototypes_proj.dtype
        )
        
        for i in range(self.total_prototypes):
            combined_prototypes[i * 2] = text_prototypes_proj[i]      # Text prototype
            combined_prototypes[i * 2 + 1] = learnable_prototypes[i]  # Learnable prototype
        
        projected_prototypes = F.normalize(combined_prototypes, dim=-1)  # (24, 512)

        # Extract 4 raw intermediate feature maps from CONCH (768-dim, before projection)
        feats_768 = self.clip_adapter.visual_intermediates_raw(x)
        while len(feats_768) < 4:
            feats_768.append(F.avg_pool2d(feats_768[-1], 2, 2))
        
        # Name the 4 raw intermediate feature maps: F1, F2, F3, F4 (768-dim)
        F1 = feats_768[0]  # [B, 768, H1, W1]
        F2 = feats_768[1]  # [B, 768, H2, W2]
        F3 = feats_768[2]  # [B, 768, H3, W3]
        F4 = feats_768[3]  # [B, 768, H4, W4]
        
        # Apply shared decoder to refine F1-F4 → R1-R4 (768-dim → 512-dim)
        R1 = self.shared_decoder(F1)  # [B, 512, H1, W1]
        R2 = self.shared_decoder(F2)  # [B, 512, H2, W2]
        R3 = self.shared_decoder(F3)  # [B, 512, H3, W3]
        R4 = self.shared_decoder(F4)  # [B, 512, H4, W4]
        
        # Input: [R1, R2, R3, R4] at various resolutions
        # Output: [P1, P2, P3, P4] with fixed resolutions and channels
        # P1: [B, 256, 28, 28], P2: [B, 128, 56, 56], P3: [B, 64, 112, 112], P4: [B, 32, 224, 224]
        pyramid_features = self.feature_pyramid([R1, R2, R3, R4])
        P1, P2, P3, P4 = pyramid_features
        
        # Extract global image embedding (CLS token) from CONCH
        # encode_image already returns [B, 512] - no projection needed
        global_image_feat = self.clip_adapter.encode_image(x, normalize=True)  # [B, 512]
        # Store global image feature for later use
        self.global_image_feature = global_image_feat


        # Project prototypes to match each pyramid level's channel dimension
        # projected_prototypes: [24, 512] -> project to [24, 256], [24, 128], [24, 64], [24, 32]
        proto_p1 = self.proto_projections[0](projected_prototypes)  # [24, 256]
        proto_p2 = self.proto_projections[1](projected_prototypes)  # [24, 128]
        proto_p3 = self.proto_projections[2](projected_prototypes)  # [24, 64]
        proto_p4 = self.proto_projections[3](projected_prototypes)  # [24, 32]
        
        # Normalize projected prototypes
        proto_p1 = F.normalize(proto_p1, dim=-1)
        proto_p2 = F.normalize(proto_p2, dim=-1)
        proto_p3 = F.normalize(proto_p3, dim=-1)
        proto_p4 = F.normalize(proto_p4, dim=-1)
        
        # Flatten spatial dimensions for CAM computation (use pyramid features P1-P4)
        # P1: [B, 256, 28, 28], P2: [B, 128, 56, 56], P3: [B, 64, 112, 112], P4: [B, 32, 224, 224]
        P1_flat = P1.permute(0, 2, 3, 1).reshape(x.size(0), -1, P1.shape[1])  # [B, 28*28, 256]
        P2_flat = P2.permute(0, 2, 3, 1).reshape(x.size(0), -1, P2.shape[1])  # [B, 56*56, 128]
        P3_flat = P3.permute(0, 2, 3, 1).reshape(x.size(0), -1, P3.shape[1])  # [B, 112*112, 64]
        P4_flat = P4.permute(0, 2, 3, 1).reshape(x.size(0), -1, P4.shape[1])  # [B, 224*224, 32]
        
        def compute_cam_pyramid(feat_map, proto_proj, logit_scale, target_size):
            """
            Compute raw CAM for a given pyramid level.
            
            Args:
                feat_map  : [B, N, C]  - flattened pyramid features (N = H*W, C = pyramid channel)
                proto_proj: [24, C]    - prototypes projected to pyramid channel dimension
                logit_scale: scalar nn.Parameter
                target_size: (H, W) tuple for reshaping
            Returns:
                cam      : [B, 24, H, W] - raw prototype‑wise similarity map
            """
            B, N, C = feat_map.shape
            H, W = target_size
            feat_norm = F.normalize(feat_map, dim=-1)             # [B, N, C]
            proto_norm = F.normalize(proto_proj, dim=-1)          # [24, C]
            # Use logit_scale so its gradients are propagated
            sim = logit_scale * (feat_norm @ proto_norm.t())      # [B, N, 24]
            cam = sim.permute(0, 2, 1).view(B, self.total_prototypes_combined, H, W)
            return cam

        # Compute raw CAMs at all 4 pyramid levels
        cam1 = compute_cam_pyramid(P1_flat, proto_p1, self.logit_scale1, (28, 28))   # [B, 24, 28, 28]
        cam2 = compute_cam_pyramid(P2_flat, proto_p2, self.logit_scale2, (56, 56))   # [B, 24, 56, 56]
        cam3 = compute_cam_pyramid(P3_flat, proto_p3, self.logit_scale3, (112, 112)) # [B, 24, 112, 112]
        cam4 = compute_cam_pyramid(P4_flat, proto_p4, self.logit_scale4, (224, 224)) # [B, 24, 224, 224]

        cls1 = F.adaptive_avg_pool2d(cam1, 1).view(x.size(0), self.total_prototypes_combined)
        cls2 = F.adaptive_avg_pool2d(cam2, 1).view(x.size(0), self.total_prototypes_combined)
        cls3 = F.adaptive_avg_pool2d(cam3, 1).view(x.size(0), self.total_prototypes_combined)
        cls4 = F.adaptive_avg_pool2d(cam4, 1).view(x.size(0), self.total_prototypes_combined)
        
        feature_map_for_diversity = P4  # Use deepest pyramid level P4 [B, 32, 224, 224]
        cam_weights = None
        text_features_out = (self.prototypes, self.class_prototypes)  # ← Both!

        if self.text_fusion is not None:
            # cam_fusion_levels = (1, 2, 3) corresponds to R2, R3, R4 (refined features)
            pooled_feats = [
                F.adaptive_avg_pool2d(R2, 1).flatten(1),
                F.adaptive_avg_pool2d(R3, 1).flatten(1),
                F.adaptive_avg_pool2d(R4, 1).flatten(1)
            ]
            proto_levels = [projected_prototypes] * 3
            cam_weights, _ = self.text_fusion(pooled_feats, proto_levels, return_text=True)
        
        contrastive_loss = None
        if labels is not None and self.training:
            # global_image_feat is already extracted and stored in self.global_image_feature
            if hasattr(self, 'global_image_feature') and self.global_image_feature is not None:
                # Use only learnable prototypes (12, 512) instead of hybrid prototypes (24, 512)
                contrastive_loss = self.compute_contrastive_loss(
                    self.global_image_feature,  # [B, 512]
                    learnable_prototypes,        # [12, 512] - only learnable image prototypes
                    labels                      # [B, num_classes]
                )
        
        # Return combined prototypes for diversity loss
        return (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4,
                projected_prototypes, self.k_list.cpu().tolist(), feature_map_for_diversity,
                cam_weights, projected_prototypes, text_features_out, contrastive_loss)     

