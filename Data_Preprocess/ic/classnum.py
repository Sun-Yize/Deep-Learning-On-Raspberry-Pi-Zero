import os
import numpy as np
from PIL import Image

path = './final/flower/normal picture/'

for item1 in os.listdir(path):
    path1 = path + item1 + '/'
    print(item1)
    t=0
    for item2 in os.listdir(path1):
        t += 1
    print(t)