import torch
import cv2
import segmentation_models_pytorch as smp
import numpy as np

# load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load("agents/vision_agent/building_unet.pth", map_location="cpu"))
model.eval()


def detect_buildings(image):

    img = cv2.resize(image, (256,256))
    img = img / 255.0

    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(tensor)

    prob_map = torch.sigmoid(pred).squeeze().numpy()

    prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))

    return prob_map