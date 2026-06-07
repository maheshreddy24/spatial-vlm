import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

class DinoVLM(nn.Module):
    def __init__(self, dino_dim: int = 384, gemma_dim: int = 2304):
        super().__init__()

        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma2-3b-mix-224",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        self.vision_encoder = torch.hub.load(
            "ywyue/FiT3D", "dinov2_small_fine"
        )

        self.connector = nn.Sequential(
            nn.Linear(dino_dim, gemma_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(gemma_dim, gemma_dim, dtype=torch.bfloat16),
        )

        self.language_model = paligemma.language_model  # Gemma2Model

        # tied to embed_tokens — no extra memory, no redundant Linear init
        self.lm_head = paligemma.lm_head

        self.processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        self.register_buffer("image_token_id", torch.tensor(257152))
    
        # gradient checkpointing — cuts activation memory ~60%
        # self.language_model.gradient_checkpointing_enable()

        del paligemma
        torch.cuda.empty_cache()

    def _build_input_embeds(self, input_ids, pixel_values):
        all_embeds = self.language_model.embed_tokens(input_ids)  # [B, 256+T, 2304]

        dino_out      = self.vision_encoder.forward_features(pixel_values.float()).to(torch.bfloat16)
        patch_tokens  = dino_out[:, 1:, :]                                   # [B, 256, 384]
        vision_embeds = self.connector(patch_tokens).to(all_embeds.dtype)    # [B, 256, 2304]

        # expand mask to embedding dim
        image_mask = (input_ids == self.image_token_id)                      # [B, 256+T]
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(all_embeds) # [B, 256+T, 2304]

        # scatter vision embeds into the image positions — graph stays intact
        vision_flat = vision_embeds.reshape(-1, vision_embeds.size(-1))      # [B*256, 2304]
        all_embeds = all_embeds.clone()
        all_embeds[image_mask] = vision_flat

        return all_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        **kwargs,
    ):
        all_embeds = self._build_input_embeds(input_ids, pixel_values)

        outputs = self.language_model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state                       # [B, 256+T, 2304]
        logits = self.lm_head(hidden_states)                           # [B, 256+T, 257216] bf16

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # cast to float32 only for the loss computation, not the full tensor
            loss = F.cross_entropy(
                shift_logits.float().view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_output(self, images, prompts, max_new_tokens=200, **kwargs):
        device = next(self.connector.parameters()).device

        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        inputs["pixel_values"] = inputs["pixel_values"]

        cur_embeds      = self._build_input_embeds(inputs["input_ids"], inputs["pixel_values"])
        attn_mask       = inputs["attention_mask"]
        past_key_values = None
        generated       = []

        # for _ in range(max_new_tokens):
        #     out = self.language_model(
        #         inputs_embeds=cur_embeds,
        #         attention_mask=attn_mask,
        #         past_key_values=past_key_values,
        #         use_cache=True,
        #     )
        #     past_key_values = out.past_key_values
        #     next_token      = self.lm_head(out.last_hidden_state[:, -1, :]).argmax(dim=-1)
        #     generated.append(next_token)
        for _ in range(max_new_tokens):
            out = self.language_model(
                inputs_embeds=cur_embeds,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            next_token = self.lm_head(out.last_hidden_state[:, -1, :]).argmax(dim=-1)
            
            # ── debug ──
            # print(f"next_token id: {next_token.item()}, decoded: '{self.processor.tokenizer.decode(next_token)}'")
            # ───────────
            
            generated.append(next_token)

            if (next_token == self.processor.tokenizer.eos_token_id).all():
                print("EOS hit, stopping")  # debug
                break

            if (next_token == self.processor.tokenizer.eos_token_id).all():
                break

            cur_embeds = self.language_model.embed_tokens(next_token.unsqueeze(1))
            attn_mask  = torch.cat([
                attn_mask,
                torch.ones(attn_mask.size(0), 1, device=device, dtype=attn_mask.dtype)
            ], dim=1)

        generated = torch.stack(generated, dim=1)
        return self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def freeze_vision(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("[INFO] Freezed ViT")

    def freeze_lm(self):
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.lm_head.weight.requires_grad = False
        print("[INFO] Freezed LM")

        
    
# # this is written just for experimentation

class SigLipVLM(nn.Module):
    def __init__(self, gemma_dim: int = 2304):
        super().__init__()

        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma2-3b-mix-224",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        # self.vision_encoder = torch.hub.load(
        #     "ywyue/FiT3D", "dinov2_small_fine"
        # )
        self.vision_encoder = paligemma.vision_tower

        self.connector = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, gemma_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(gemma_dim, gemma_dim, dtype=torch.bfloat16),
        )

        self.language_model = paligemma.language_model  # Gemma2Model

        # tied to embed_tokens — no extra memory, no redundant Linear init
        self.lm_head = paligemma.lm_head

        self.processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        self.register_buffer("image_token_id", torch.tensor(257152))
    
        # gradient checkpointing — cuts activation memory ~60%
        # self.language_model.gradient_checkpointing_enable()

        del paligemma
        torch.cuda.empty_cache()

    def _build_input_embeds(self, input_ids, pixel_values):
        all_embeds = self.language_model.embed_tokens(input_ids)  # [B, 256+T, 2304]

        # dino_out      = self.vision_encoder.forward_features(pixel_values.float()).to(torch.bfloat16)
        patch_tokens = self.vision_encoder(pixel_values.float())['last_hidden_state'].to(torch.bfloat16)
        # patch_tokens  = dino_out[:, 1:, :]                                   # [B, 256, 384]
        vision_embeds = self.connector(patch_tokens).to(all_embeds.dtype)    # [B, 256, 2304]

        # expand mask to embedding dim
        image_mask = (input_ids == self.image_token_id)                      # [B, 256+T]
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(all_embeds) # [B, 256+T, 2304]

        # scatter vision embeds into the image positions — graph stays intact
        vision_flat = vision_embeds.reshape(-1, vision_embeds.size(-1))      # [B*256, 2304]
        all_embeds = all_embeds.clone()
        all_embeds[image_mask] = vision_flat

        return all_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        **kwargs,
    ):
        all_embeds = self._build_input_embeds(input_ids, pixel_values)

        outputs = self.language_model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state                       # [B, 256+T, 2304]
        logits = self.lm_head(hidden_states)                           # [B, 256+T, 257216] bf16

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # cast to float32 only for the loss computation, not the full tensor
            loss = F.cross_entropy(
                shift_logits.float().view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # @torch.no_grad()
    # def generate_output(self, images, prompts, max_new_tokens=200, **kwargs):
    #     device = next(self.connector.parameters()).device

    #     inputs = self.processor(
    #         images=images,
    #         text=prompts,
    #         return_tensors="pt",
    #         padding=True,
    #     ).to(device)

    #     inputs["pixel_values"] = inputs["pixel_values"]

    #     cur_embeds      = self._build_input_embeds(inputs["input_ids"], inputs["pixel_values"])
    #     attn_mask       = inputs["attention_mask"]
    #     past_key_values = None
    #     generated       = []

    #     for _ in range(max_new_tokens):
    #         out = self.language_model(
    #             inputs_embeds=cur_embeds,
    #             attention_mask=attn_mask,
    #             past_key_values=past_key_values,
    #             use_cache=True,
    #         )
    #         past_key_values = out.past_key_values
    #         next_token = self.lm_head(out.last_hidden_state[:, -1, :]).argmax(dim=-1)

    #         generated.append(next_token)

    #         if (next_token == self.processor.tokenizer.eos_token_id).all():
    #             print("EOS hit, stopping")  # debug
    #             break

    #         if (next_token == self.processor.tokenizer.eos_token_id).all():
    #             break

    #         cur_embeds = self.language_model.embed_tokens(next_token.unsqueeze(1))
    #         attn_mask  = torch.cat([
    #             attn_mask,
    #             torch.ones(attn_mask.size(0), 1, device=device, dtype=attn_mask.dtype)
    #         ], dim=1)

    #     generated = torch.stack(generated, dim=1)
    #     return self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    @torch.no_grad()
    def generate_output(self, images, prompts, max_new_tokens=200):
        device = next(self.connector.parameters()).device
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)

        input_embeds = self._build_input_embeds(inputs["input_ids"], inputs["pixel_values"])

        output_ids = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def freeze_vision(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("[INFO] Freezed ViT")

    def freeze_lm(self):
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.lm_head.weight.requires_grad = False
        print("[INFO] Freezed LM")