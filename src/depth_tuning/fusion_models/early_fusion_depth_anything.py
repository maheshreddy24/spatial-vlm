import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2




class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()

        self.q = nn.Linear(d_model, head_size)
        self.k = nn.Linear(d_model, head_size)
        self.v = nn.Linear(d_model, head_size)

    def forward(self, ve_feat, da_feat):
        """
        ve_feat : [B, N_v, D]
        da_feat : [B, N_d, D]
        """

        Q = self.q(ve_feat)  # [B, N_v, H]
        K = self.k(da_feat)  # [B, N_d, H]
        V = self.v(da_feat)  # [B, N_d, H]

        attn = Q @ K.transpose(-2, -1)
        attn = attn / (Q.shape[-1] ** 0.5)

        attn = torch.softmax(attn, dim=-1)

        out = attn @ V  # [B, N_v, H]

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.head_size = d_model // n_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(d_model, self.head_size)
                for _ in range(n_heads)
            ]
        )

        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, ve_feat, da_feat):

        head_outputs = [
            head(ve_feat, da_feat)
            for head in self.heads
        ]

        out = torch.cat(head_outputs, dim=-1)
        out = self.Wo(out)

        return out

class CrossAttn(nn.Module):
    def __init__(
        self,
        ve_size,
        da_size,       # fixed: was da_feat_size (typo in original)
        n_heads=8
    ):
        super().__init__()
        self.da_proj = nn.Linear(da_size, ve_size)
        self.cross_attn = MultiHeadAttention(d_model=ve_size, n_heads=n_heads)
        self.norm = nn.LayerNorm(ve_size)
        self.A = nn.Parameter(torch.zeros(1, 1, ve_size, dtype=torch.float32), requires_grad=True)

    def forward(self, ve_feat, da_feat):
        da_feat = self.da_proj(da_feat)           # [B, 1369, ve_size]
        attn_out = self.cross_attn(ve_feat, da_feat)
        out = self.norm(ve_feat + self.A * attn_out)
        return out


class EarlyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits'

        self.depth_anything = DepthAnythingV2(**model_configs[encoder])
        self.depth_anything.load_state_dict(
            torch.load('Depth_Anything_V2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
        )
        self.depth_anything = self.depth_anything.to(self.device).eval()

        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma2-3b-mix-224",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        self.vision_encoder = self.paligemma.vision_tower
        ve_size = self.paligemma.vision_tower.config.hidden_size  # e.g. 1152
        da_size = 384  # vits patch embedding dim

        self.cross_attn = CrossAttn(ve_size=ve_size, da_size=da_size)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        pixel_values  : [B, 3, H, W]  — same image fed to both encoders
        input_ids     : [B, seq_len]
        attention_mask: [B, seq_len]
        labels        : [B, seq_len]  — for training; None at inference
        """

        # depth_anything.forward returns (depth_map, features)
        # feat[0][0] : [B, 1369, 384]  — patch-level embeddings
        # feat[0][1] : [B, 384]        — CLS token (not used here)
        with torch.no_grad():
            _, feat = self.depth_anything.forward(pixel_values)
        da_feat = feat[0][0].to(self.device)          # [B, 1369, 384]
        da_feat = da_feat.to(torch.bfloat16)           # match PaliGemma dtype

        # PaliGemma's SigLIP tower returns [B, 256, ve_size]
        ve_out = self.vision_encoder(pixel_values)     # BaseModelOutput
        ve_feat = ve_out.last_hidden_state             # [B, 256, ve_size]

        # ── 3. Early Fusion: cross-attend Ve (query) over DA (key/value) ─────
        fused = self.cross_attn(ve_feat, da_feat)      # [B, 256, ve_size]

        # ── 4. Inject fused tokens into PaliGemma's multi-modal projector ────
        # The projector maps vision tokens → language embedding space
        projected = self.paligemma.multi_modal_projector(fused)  # [B, 256, hidden]

        # ── 5. Build full token embeddings (text + fused vision) ─────────────
        text_embeds = self.paligemma.language_model.model.embed_tokens(input_ids)

        # PaliGemma convention: image tokens occupy the first 256 positions
        # Replace them with our fused+projected tokens
        inputs_embeds = text_embeds.clone()
        inputs_embeds[:, :projected.shape[1], :] = projected

        # ── 6. Forward through LLM backbone ──────────────────────────────────
        outputs = self.paligemma.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs  # CausalLMOutputWithPast; .loss available when labels given
    
    