
import colorsys
import json
import os
from pathlib import Path
import random
import cv2
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import h5py
import os

import torch
import torchvision
import numpy as np
from PIL import Image



def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors