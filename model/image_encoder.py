import torch.nn as nn
from transformers import CLIPModel, AutoProcessor


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.linear_layers = nn.Sequential( nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_proc(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.clip_model.get_image_features(pixel_values=x)
        x = self.linear_layers(x)
        return x