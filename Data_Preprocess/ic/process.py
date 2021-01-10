import os
import random
import shutil
import string
from PIL import Image


def IsValidImage(img_path):
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path):
    if IsValidImage(img_path):
        try:
            str = img_path.rsplit(".", 1)
            output_img_path = str[0] + ".jpg"
            im = Image.open(img_path)
            im.save(output_img_path)
            return True
        except:
            return False
    else:
        return False


path = 'usedata/fruit/normal picture/'
finalpath = 'final/fruit/normal picture/'
for items1 in os.listdir(path):
    if items1 != 'apple':
        continue
    print(items1)
    path1 = path + items1 + '/'
    path_final = finalpath + items1 + '/'
    if not os.path.exists(path_final):
        os.mkdir(path_final)
    if os.listdir(path_final):
        shutil.rmtree(path_final)
        os.mkdir(path_final)
    i = 0
    for items2 in os.listdir(path1):
        print(items2)
        if items2 == 'cir':
            continue
        path2 = path1 + items2 + '/'
        for items3 in os.listdir(path2):
            path3 = path2 + items3 + '/'
            for items4 in os.listdir(path3):
                img_path = path3 + items4
                transimg(img_path)
                a = ''.join(random.choices(string.ascii_lowercase, k=20))
                os.rename(img_path, path3+a+'.jpg')
                shutil.copy(path3+a+'.jpg', path_final)
                os.rename(path_final+a+'.jpg', path_final+str(i)+'.jpg')
                i += 1
#
# import os
# import random
# import shutil
# import string
# from PIL import Image
#
#
# def IsValidImage(img_path):
#     bValid = True
#     try:
#         Image.open(img_path).verify()
#     except:
#         bValid = False
#     return bValid
#
#
# def transimg(img_path):
#     if IsValidImage(img_path):
#         try:
#             str = img_path.rsplit(".", 1)
#             output_img_path = str[0] + ".jpg"
#             im = Image.open(img_path)
#             im.save(output_img_path)
#             return True
#         except:
#             return False
#     else:
#         return False
#
#
# path = 'usedata/flower/normal picture/'
# finalpath = 'final/flower/normal picture/'
# for items1 in os.listdir(path):
#     if items1 != 'veronica':
#         continue
#     print(items1)
#     path1 = path + items1 + '/'
#     path_final = finalpath + items1 + '/'
#     if not os.path.exists(path_final):
#         os.mkdir(path_final)
#     if os.listdir(path_final):
#         shutil.rmtree(path_final)
#         os.mkdir(path_final)
#     i = 0
#     for items3 in os.listdir(path1):
#         path3 = path1 + items3 + '/'
#         for items4 in os.listdir(path3):
#             img_path = path3 + items4
#             transimg(img_path)
#             a = ''.join(random.choices(string.ascii_lowercase, k=20))
#             os.rename(img_path, path3+a+'.jpg')
#             shutil.copy(path3+a+'.jpg', path_final)
#             os.rename(path_final+a+'.jpg', path_final+str(i)+'.jpg')
#             i += 1
#
