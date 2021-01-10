import os
import random
import shutil

i1 = 15
path = './final/flower/normal picture/'
newpath = './fruit_flower/train/'
testpath = './fruit_flower/test/'
if not os.path.exists(newpath):
    os.mkdir(newpath)
if not os.path.exists(testpath):
    os.mkdir(testpath)

for items1 in os.listdir(path):
    path1 = path + items1 + '/'
    newpath1 = newpath + str(i1) + '/'
    testpath1 = testpath + str(i1) + '/'
    if not os.path.exists(newpath1):
        os.mkdir(newpath1)
    if os.listdir(newpath1):
        shutil.rmtree(newpath1)
        os.mkdir(newpath1)
    if not os.path.exists(testpath1):
        os.mkdir(testpath1)
    if os.listdir(testpath1):
        shutil.rmtree(testpath1)
        os.mkdir(testpath1)
    i1 += 1
    i = 0
    for items2 in os.listdir(path1):
        i += 1
    resultList = random.sample(range(0, i), 1100)
    for t in range(1100):
        if t < 1000:
            shutil.copy(path1 + str(resultList[t]) + '.jpg', newpath1)
            os.rename(newpath1 + str(resultList[t]) + '.jpg', newpath1 + 'img' + str(t+1) + '.jpg')
        else:
            shutil.copy(path1 + str(resultList[t]) + '.jpg', testpath1)
            os.rename(testpath1 + str(resultList[t]) + '.jpg', testpath1 + 'img' + str(t + 1) + '.jpg')
    # print(path1 + str(resultList[1]) + '.jpg')
    # print(newpath1)
    # shutil.copy(path1 + str(resultList[1]) + '.jpg', newpath1)


