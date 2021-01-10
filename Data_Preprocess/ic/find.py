import os
import numpy as np
from PIL import Image

path = './fruit_flower/train/'

for item1 in os.listdir(path):
    path1 = path + item1 + '/'
    print(item1)
    for item2 in os.listdir(path1):
        path2 = path1 + item2
        if os.path.getsize(path2) == 0:
            print(path2)
        else:
            img = Image.open(path2)
            # print(img_array)
            # if np.shape(img_array.shape)[0] != 3:
            #     print('p', path2)
            if img.size[0] < 224 or img.size[1] < 224:
                print('q', path2)
