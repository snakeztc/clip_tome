from tome.clip.clip_vit import load
from PIL import Image
import torch
from tome.patch.oai_clip import apply_patch

if __name__ == "__main__":
    device = 'cpu'
    model, _ = load("/Users/tonyzhao/Documents/projects/ToMe/ViT-B-16.pt", device=torch.device(device), jit=False,
                    load_pretrain_weights=True)
    model = model.eval()
    apply_patch(model)
    model.r = 16

    a = Image.open('/Users/tonyzhao/Documents/projects/ToMe/examples/images/husky.png').convert('RGB')
    image_input = [
        _(x).unsqueeze(0).to(device).half() for x in [a]
    ] * 2
    image_input = torch.cat(image_input, dim=0)
    image_features = model(image_input)
    print(image_features)
    print(image_features.shape)