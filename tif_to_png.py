import os
import tif as tif
import glob
from PIL import Image

files = glob.glob(os.path.join("recordings", "session_2023_11_25-12_21_31", "video_images", "*.tif"))

for file in files:
    img = Image.open(file)
    img.save(file.split('.')[0] + ".png")
    os.remove(file)