import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType


import torch
import torch.nn as nn
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class Model(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        model = AutoModel.from_pretrained(
            'OpenGVLab/InternVL2_5-2B',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,   # <-- add this
        ).eval()

        self.base = model.vision_model
        model = None
        for p in self.base.parameters():
            p.requires_grad = False

        total_params = sum([p.numel() for p in self.base.parameters()])
        print(f'total parameters for base models: {total_params}')
        self.base.gradient_checkpointing_disable()

        hidden_size = self.base.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()        # always keep backbone in eval mode
        return self

    def forward(self, image):
        with torch.no_grad():
            outputs = self.base(image)
        cls_token = outputs.last_hidden_state[:, 0].float()
        logits = self.classifier(cls_token)
        return logits

    
class DinoWrapper(nn.Module):
    def __init__(self, num_classes,model_name = 'facebook/dinov2-large'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

        for p in self.model.parameters():
            p.requires_grad = False

        hidden_dim = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            params =  self.model(x).last_hidden_state  # [B, 197, 1024]
        emb = params[:, 0, :]
        logits = self.classifier(emb)
        return logits

class ModelPeft(nn.Module):

    def __init__(self, num_classes, target_modules):
        super().__init__()

        model = AutoModel.from_pretrained(
            'OpenGVLab/InternVL2_5-2B',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval()

        self.base = model.vision_model
        model = None

        for p in self.base.parameters():
            p.requires_grad = False

        hidden_size = self.base.config.hidden_size

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            # Remove task_type — it causes PEFT to assume an NLP model
            # task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.model_peft = get_peft_model(self.base, lora_config)
        self.model_peft.print_trainable_parameters()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, image):
        # Pass pixel_values explicitly — InternVisionModel expects this, not input_ids
        outputs = self.model_peft(pixel_values=image)
        cls_token = outputs.last_hidden_state[:, 0, :].float()
        logits = self.classifier(cls_token)
        return logits.float()