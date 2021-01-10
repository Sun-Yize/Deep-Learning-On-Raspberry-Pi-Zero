import os
import shutil


ori1 = './newimg/flower/'
newimg = './newimg/combine/'
t = 0
if not os.path.exists(newimg):
    os.mkdir(newimg)
if os.listdir(newimg):
    shutil.rmtree(newimg)
    os.mkdir(newimg)
for items1 in os.listdir(ori1):
    path1 = ori1 + items1 + '/'
    i = 0
    for items2 in os.listdir(path1):
        i += 1
    print(int((i/2)+1))
    for p in range(1, int((i/2)+1)):
        shutil.copy(path1 + str(p) + '.jpg', newimg)
        shutil.copy(path1 + str(p) + '.xml', newimg)
        os.rename(newimg + str(p) + '.jpg', newimg + 'img' + str(t) + '.jpg')
        os.rename(newimg + str(p) + '.xml', newimg + 'img' + str(t) + '.xml')
        t += 1
