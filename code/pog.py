import numpy as np
import cv2
import pyvirtualcam
import pkgutil

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)