import os
import sys

import torch
from torch import nn
import vgg
from torchvision.transforms import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        if device.type == 'cuda' :
            checkpoint = torch.load(resume_path)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        del checkpoint
    else:
        sys.exit("No checkpoint found at '{}'".format(resume_path))
    return model


def encode(model, img):
    with torch.no_grad():
        code = model.module.encoder(img).cpu().numpy()
    return code

def get_model():
    model = vgg.VGGAutoEncoder(vgg.get_configs())
    model = nn.DataParallel(model).to(device)

    return model


def main(image_path):
    # Load model
    model = get_model()
    load_dict("imagenet-vgg16.pth", model)

    # Transform image to match model input size
    trans = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])
    img = Image.open(image_path).convert("RGB")
    img = trans(img).unsqueeze(0).to(device)

    # Run image trough model
    model.eval()

    code = encode(model, img)

    print(code.shape)
    return code

if __name__ == '__main__':
    main(sys.argv[1])