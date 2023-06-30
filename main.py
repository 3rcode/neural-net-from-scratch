import os
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_train = os.path.join(ROOT_DIR, 'data', 'train', 'cats')
height = width = 128
channel = 3

images = []
for root, dirs, files in os.walk(data_train):
    for file in files:
        images.append(os.path.join(root, file))


def get_rgb(image_path):
    global height, width, channel
    image = Image.open(image_path).resize((height, width))
    rgb = np.array(image.getdata()).reshape((height, width, channel))
    return rgb


if __name__ == '__main__':
    pass