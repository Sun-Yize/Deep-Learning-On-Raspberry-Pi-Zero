import os
import numpy as np
from PIL import Image

path = './final/flower/normal picture/'

for item1 in os.listdir(path):
    # if item1 != 'apple':
    #     continue
    path1 = path + item1 + '/'
    t = 0
    print(t)
    for item2 in os.listdir(path1):
        path2 = path1 + item2
        # if os.path.getsize(path2) == 0:
        #     os.remove(path2)
        #     print(path2)
        # else:
        #     # img = Image.open(path2)
        #     # img_array = np.array(img)
        #     # if np.shape(img_array.shape)[0] != 3:
        #     #     os.remove(path2)
        #     #     print('p', path2)
        #     # else:
        os.rename(path2, path1+str(t)+'.jpg')
        t += 1


