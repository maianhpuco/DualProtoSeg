from typing import Dict, Iterable, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

_tokenizer = get_tokenizer()

class ConchPromptLearner(nn.Module):
    """
    CoOp-style Prompt Learner for CONCH
    - Each class gets its own unique learnable prompt context
    - Best performance on pathology tasks (used in top 2025 papers)
    """
    def __init__(
        self,
        class_prompts: Dict[int, List[str]],
        model_conch,
        n_ctx: int = 16,
        class_token_position: Literal["front", "middle", "end"] = "end",
        ctx_init: str = "a histopathology image of",
        device: str = "cuda"
    ):
        super().__init__()
        self.n_cls = len(class_prompts)
        self.n_ctx = n_ctx
        self.position = class_token_position
        self.device = torch.device(device)
        self.model_conch = model_conch #.to(self.device)

        # === 1. Extend vocabulary with n_ctx × n_cls new soft tokens ===
        old_embedding = model_conch.text.token_embedding
        old_num_tokens = old_embedding.num_embeddings
        embed_dim = old_embedding.embedding_dim  # 768
        print("old_embedding:", old_embedding)
        print("old_num_tokens:", old_num_tokens) 
        print("embed_dim:", embed_dim)
        print("---")
        tokens_per_class = n_ctx
        total_new_tokens = tokens_per_class * self.n_cls
        self.soft_token_start_ids = []  # Start ID for each class
        print("total new token", total_new_tokens)
        
        new_num_tokens = old_num_tokens + total_new_tokens
        new_embedding = nn.Embedding(new_num_tokens, embed_dim)
        new_embedding = new_embedding.to(self.device)
        new_embedding.weight.data[:old_num_tokens] = old_embedding.weight.data
        model_conch.text.token_embedding = new_embedding
        print("new_embedding.shape", new_embedding.num_embeddings) 
        # === 2. Initialize each class's tokens from text ===
        init_tokens = torch.tensor(_tokenizer.encode(ctx_init), device=self.device)
        if len(init_tokens) < n_ctx + 2:
            pad = init_tokens[-1:].repeat(n_ctx + 2 - len(init_tokens))
            init_tokens = torch.cat([init_tokens, pad])

        with torch.no_grad():
            init_emb = model_conch.text.token_embedding(init_tokens.unsqueeze(0))
            base_init = init_emb[0, 1:1 + n_ctx]  # (16, 768)

        # === 3. Create class-specific learnable tokens ===
        ctx_vectors = []
        start_id = old_num_tokens

        for cls_idx in range(self.n_cls):
            # Optional: add small noise per class for diversity
            noise = torch.randn_like(base_init) * 0.02
            class_init = base_init + noise
            # Copy to embedding table
            model_conch.text.token_embedding.weight.data[start_id:start_id + n_ctx].copy_(class_init)
            ctx_vectors.append(class_init)
            self.soft_token_start_ids.append(start_id)
            start_id += n_ctx

        # Stack → (n_cls, n_ctx, 768)
        self.ctx = nn.Parameter(torch.stack(ctx_vectors))  # ← TRAINABLE!
        print("leanable part: ", self.ctx.shape) 
        # === 4. Tokenize class prompts ===
        self.input_ids_list = []
        self.num_per_class = []

        for prompts in class_prompts.values():
            self.num_per_class.append(len(prompts))
            for text in prompts:
                tokens = torch.tensor(_tokenizer.encode(text), dtype=torch.long, device=self.device)
                self.input_ids_list.append(tokens)

        self.prompts_per_class = list(class_prompts.values())  # ← This is what ClsNetwork needs!

        self.total_prompts = len(self.input_ids_list) # ← 12

    def forward(self):
        all_input_ids = []
        prompt_idx = 0

        for cls_idx in range(self.n_cls):
            n_p = self.num_per_class[cls_idx]
            start_id = self.soft_token_start_ids[cls_idx]
            soft_ids = torch.arange(start_id, start_id + self.n_ctx, device=self.device)

            self.model_conch.text.token_embedding.weight.data[soft_ids] = self.ctx[cls_idx]

            for i in range(n_p):
                orig = self.input_ids_list[prompt_idx + i]
                bos = orig[0:1]
                class_tokens = orig[1:-1]
                eos = orig[-1:]

                if self.position == "end":
                    seq = [bos, soft_ids, class_tokens, eos]
                elif self.position == "front":
                    seq = [bos, class_tokens, eos, soft_ids]
                elif self.position == "middle":
                    half = self.n_ctx // 2
                    seq = [bos, soft_ids[:half], class_tokens, eos, soft_ids[half:]]

                input_ids = torch.cat(seq)
                if input_ids.shape[0] > 77:
                    input_ids = input_ids[:77]
                else:
                    pad = torch.zeros(77 - len(input_ids), dtype=torch.long, device=self.device)
                    input_ids = torch.cat([input_ids, pad])

                all_input_ids.append(input_ids)
            prompt_idx += n_p

        return torch.stack(all_input_ids)

    def get_text_features(self):
        input_ids = self()
        
        # Build combined embedding weight: frozen base + learnable ctx
        # This creates a proper computation graph where gradients flow through self.ctx
        base_weight = self.model_conch.text.token_embedding.weight  # Frozen base
        
        # Construct weight matrix with learnable parts
        # We'll use torch.cat to combine, which preserves gradients
        weight_parts = []
        prev_end = 0
        
        for cls_idx in range(self.n_cls):
            start_id = self.soft_token_start_ids[cls_idx]
            end_id = start_id + self.n_ctx
            
            # Add frozen part before this learnable section
            if start_id > prev_end:
                weight_parts.append(base_weight[prev_end:start_id])
            
            # Add learnable part (this preserves gradient connection to self.ctx!)
            weight_parts.append(self.ctx[cls_idx])
            
            prev_end = end_id
        
        # Add remaining frozen part
        if prev_end < base_weight.shape[0]:
            weight_parts.append(base_weight[prev_end:])
        
        # Concatenate all parts - this creates a computation graph
        combined_weight = torch.cat(weight_parts, dim=0)
        
        # Use functional embedding with combined weight
        # This preserves gradients because combined_weight includes self.ctx
        embeddings = F.embedding(input_ids, combined_weight)
        
        # Now we need to manually do the text encoder forward pass
        # Get the text encoder components
        text_encoder = self.model_conch.text
        
        # Add positional embeddings
        seq_len = embeddings.shape[1]
        pos_emb = text_encoder.positional_embedding[:seq_len]
        x = embeddings + pos_emb.unsqueeze(0)
        
        # Permute for transformer: [seq_len, batch, embed_dim]
        x = x.permute(1, 0, 2)
        
        # Forward through transformer (frozen, but gradients flow through activations)
        # Don't use no_grad here - we need gradients to flow through to embeddings
        x = text_encoder.transformer(x)
        
        # Permute back: [batch, seq_len, embed_dim]
        x = x.permute(1, 0, 2)
        
        # Get EOT token position (last non-padding token)
        # For simplicity, use the last position
        eot_pos = seq_len - 1
        text_features = x[:, eot_pos, :]
        
        # Apply layer norm and projection (frozen, but gradients flow through)
        text_features = text_encoder.ln_final(text_features)
        text_features = text_features @ text_encoder.text_projection
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # print("text_features.shape", text_features.shape)
        
        text_features = text_features.float()

        splits = torch.split(text_features, self.num_per_class)
        # print("total: ", len(splits), "- each - ", splits[0].shape)
        class_features = torch.stack([s.mean(dim=0) for s in splits])
        return text_features, class_features  # (n_cls, 768) 
    

class ConchAdapter(nn.Module):
    def __init__(
        self,
        model_name: str = "conch_ViT-B-16",
        checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
        class_prompts: Dict[int, List[str]] = None,      # REQUIRED
        prompt_n_ctx: int = 16,
        prompt_position: Literal["front", "middle", "end"] = "end",
        freeze_conch: bool = True,
        hf_hub: str = "MahmoodLab/conch",
        force_image_size: Optional[int] = 224,
        proj_contrast: bool = False, 
        **kwargs
    ):
        super().__init__()
        if class_prompts is None:
            raise ValueError("class_prompts is required – this adapter always uses learned prompts")

        self.device = device

        # Load CONCH
        self.model, self.preprocess = create_model_from_pretrained(
            model_name,
            checkpoint_path=checkpoint_path or hf_hub,
            device=device,
            **kwargs
        )
        visual = getattr(self.model, "visual", None)
        self.image_size = getattr(visual, "image_size", force_image_size)
        self.image_mean = getattr(visual, "image_mean", (0.5, 0.5, 0.5))
        self.image_std = getattr(visual, "image_std", (0.5, 0.5, 0.5)) 
        self.proj_contrast = proj_contrast 
        
        # ALWAYS use learned prompts
        self.prompt_learner = ConchPromptLearner(
            class_prompts=class_prompts,
            model_conch=self.model,
            n_ctx=prompt_n_ctx,
            class_token_position=prompt_position,
            ctx_init="a histopathology image of",
            device=device
        )

        self.embed_dim = self.model.text.text_projection.shape[1]
        self.image_size = 224 
        
        # This projects 768-dim CONCH visual features to 512-dim for compatibility
        visual_dim = 768  # CONCH ViT-B-16 embedding dimension
        proj_dim = 512    # Target dimension for feature maps
        self.visual_proj = nn.Linear(visual_dim, proj_dim)
        nn.init.eye_(self.visual_proj.weight[:min(visual_dim, proj_dim), :min(visual_dim, proj_dim)])
        nn.init.constant_(self.visual_proj.bias, 0)
        
        # Freeze CONCH, only train prompts and visual projection
        if freeze_conch:
            for p in self.model.parameters():
                p.requires_grad_(False)
            # Make prompt learner's ctx trainable
            self.prompt_learner.ctx.requires_grad_(True)
            
            # This allows gradients to flow through embedding lookup
            for cls_idx in range(self.prompt_learner.n_cls):
                start_id = self.prompt_learner.soft_token_start_ids[cls_idx]
                # Make these embedding rows trainable (they'll be updated by ctx)
                self.model.text.token_embedding.weight[start_id:start_id + self.prompt_learner.n_ctx].requires_grad_(True)
            
            # Make visual_proj trainable
            self.visual_proj.requires_grad_(True)

        self.model.eval()

    def encode_text(self, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Always uses learned CoCoOp prompts → returns both levels"""
        text_features, class_features = self.prompt_learner.get_text_features()
        
        # print("text_features.shape", text_features.shape)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
            class_features = F.normalize(class_features, dim=-1)
            
        return text_features, class_features
    
    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        proj = self.proj_contrast
        with torch.no_grad():
            feats = self.model.encode_image(images.to(self.device), normalize=normalize, proj_contrast=proj)
        return feats

    # Multi-scale features
    def visual_intermediates(self, images: torch.Tensor) -> List[torch.Tensor]:
        images = images.to(self.device)
        # visual_proj is now registered in __init__, no need to create it here
        
        def _tokens_to_map(tok: torch.Tensor) -> Optional[torch.Tensor]:
            # tok: [B, L, C] with CLS at position 0
            b, l, c = tok.shape
            side = int((l - 1) ** 0.5)
            if side * side != (l - 1):
                return None
            projected = self.visual_proj(tok) # [B, L, 512] 
            patch = projected[:, 1:, :].permute(0, 2, 1).reshape(b, 512, side, side) 
            # patch = tok[:, 1:, :].permute(0, 2, 1).reshape(b, c, side, side)
            return patch

        def _collect_vit_blocks(target_idxs: List[int]) -> List[torch.Tensor]:
            trunk = getattr(getattr(self.model, "visual", None), "trunk", None)
            if trunk is None or not hasattr(trunk, "blocks"):
                return []
            outputs: Dict[int, torch.Tensor] = {}
            hooks = []
            for idx in target_idxs:
                if idx >= len(trunk.blocks):
                    continue
                hooks.append(
                    trunk.blocks[idx].register_forward_hook(lambda _, __, out, idx=idx: outputs.__setitem__(idx, out))
                )
            try:
                _ = trunk(images)
            finally:
                for h in hooks:
                    h.remove()
            return [outputs[i] for i in sorted(outputs.keys()) if i in outputs]

        # 0-indexed block taps for a 12-layer ViT-B
        target_layers = [2, 5, 8, 11]
        # The CONCH model is frozen anyway, so this won't compute gradients for it,
        # but it allows gradients to flow through visual_proj to the loss
        block_tokens = _collect_vit_blocks(target_layers)

        # Fallback if no block captures succeeded
        if not block_tokens:
            _, tokens = self.model._encode_image(images, normalize=False)
            fmap = _tokens_to_map(tokens) if tokens is not None else None
            return [fmap] if fmap is not None else []

        maps = [m for m in (_tokens_to_map(tok) for tok in block_tokens) if m is not None]
        if not maps:
            return []
        while len(maps) < 4:
            maps.append(maps[-1])
        return maps[:4]

    def visual_intermediates_raw(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Return raw intermediate CONCH feature maps (768-dim, before projection).
        Used when we want to apply custom refinement (e.g., LightweightFPNDecoder).
        
        Returns:
            List of 4 feature maps: [B, 768, H, W] each
        """
        images = images.to(self.device)
        
        def _tokens_to_map_raw(tok: torch.Tensor) -> Optional[torch.Tensor]:
            # tok: [B, L, C] with CLS at position 0, C=768
            b, l, c = tok.shape
            side = int((l - 1) ** 0.5)
            if side * side != (l - 1):
                return None
            # Return raw 768-dim features without projection
            patch = tok[:, 1:, :].permute(0, 2, 1).reshape(b, c, side, side)  # [B, 768, side, side]
            return patch

        def _collect_vit_blocks(target_idxs: List[int]) -> List[torch.Tensor]:
            trunk = getattr(getattr(self.model, "visual", None), "trunk", None)
            if trunk is None or not hasattr(trunk, "blocks"):
                return []
            outputs: Dict[int, torch.Tensor] = {}
            hooks = []
            for idx in target_idxs:
                if idx >= len(trunk.blocks):
                    continue
                hooks.append(
                    trunk.blocks[idx].register_forward_hook(lambda _, __, out, idx=idx: outputs.__setitem__(idx, out))
                )
            try:
                _ = trunk(images)
            finally:
                for h in hooks:
                    h.remove()
            return [outputs[i] for i in sorted(outputs.keys()) if i in outputs]

        # 0-indexed block taps for a 12-layer ViT-B
        target_layers = [2, 5, 8, 11]
        block_tokens = _collect_vit_blocks(target_layers)

        # Fallback if no block captures succeeded
        if not block_tokens:
            _, tokens = self.model._encode_image(images, normalize=False)
            fmap = _tokens_to_map_raw(tokens) if tokens is not None else None
            return [fmap] if fmap is not None else []

        maps = [m for m in (_tokens_to_map_raw(tok) for tok in block_tokens) if m is not None]
        if not maps:
            return []

        # Pad/trim to 4 levels
        while len(maps) < 4:
            maps.append(maps[-1])
        maps = maps[:4]

        # Return raw 768-dim feature maps (no interpolation/pooling)
        return maps