import os
import random
import shutil
import string


ori1 = './imgnet/fruit/'
ori2 = './newimg/fruit/'
for items1 in os.listdir(ori1):
    path1 = ori1 + items1 + '/'
    newimg = ori2 + items1 + '/'
    if not os.path.exists(newimg):
        os.mkdir(newimg)
    if os.listdir(newimg):
        shutil.rmtree(newimg)
        os.mkdir(newimg)
    for items2 in os.listdir(path1):
        path = path1 + items2 + '/Annotation/' + items2 + '/'
        imgpath = path1 + items2 + '/'
        t = 1
        for xmls in os.listdir(path):
            if not os.path.exists(imgpath + xmls.split(sep='.')[0] + '.JPEG'):
                continue
            shutil.copy(path + xmls, newimg)
            os.rename(newimg + xmls, newimg + str(t) + '.xml')
            shutil.copy(imgpath + xmls.split(sep='.')[0] + '.JPEG', newimg)
            os.rename(newimg + xmls.split(sep='.')[0] + '.JPEG', newimg + str(t) + '.jpg')
            t += 1
