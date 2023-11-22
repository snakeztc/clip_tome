import timm, tome
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

if __name__ == "__main__":
    device = 'cpu'
    model_name = "vit_base_patch16_224"
    # Load a pretrained model, can be any vit / deit model.
    model = timm.create_model(model_name, pretrained=True)
    # Patch the model with ToMe.
    tome.patch.timm(model)
    # Set the number of tokens reduced per layer. See paper for details.
    model.r = 16
    img = Image.open("examples/images/husky.png")
    input_size = model.default_cfg["input_size"][1]

    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])

    img_tensor = transform(img)[None, ...]

    x = model(img_tensor)
    print(x)