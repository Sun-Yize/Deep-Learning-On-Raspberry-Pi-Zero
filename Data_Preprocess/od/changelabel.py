# import os
# import xml.etree.ElementTree as ET
#
# #程序功能：批量修改VOC数据集中xml标签文件的标签名称
# def changelabelname(inputpath):
#     listdir = os.listdir(inputpath)
#     for file in listdir:
#         if file.endswith('xml'):
#             file = os.path.join(inputpath,file)
#             tree = ET.parse(file)
#             root = tree.getroot()
#             for object1 in root.findall('object'):
#                 for sku in object1.findall('name'):           #查找需要修改的名称
#                     if (sku.text == 'stawberry'):               #‘preName’为修改前的名称
#                         sku.text = 'strawberry'                 #‘TESTNAME’为修改后的名称
#                         tree.write(file,encoding='utf-8')     #写进原始的xml文件并避免原始xml中文字符乱码
#                     else:
#                         pass
#         else:
#             pass
#
# if __name__ == '__main__':
#     inputpath = './source/tracking_fruits_dataset/train_dataset/VOC2007/combine'  #此处替换为自己的路径
#     changelabelname(inputpath)


# import os
# import xml.etree.ElementTree as ET
#
# #程序功能：批量修改VOC数据集中xml标签文件的标签名称
# def changelabelname(inputpath):
#     listdir = os.listdir(inputpath)
#     for file in listdir:
#         if file.endswith('xml'):
#             file = os.path.join(inputpath,file)
#             tree = ET.parse(file)
#             root = tree.getroot()
#             for object1 in root.findall('filename'):
#                 object1.text = object1.text + '.jpg'
#                 tree.write(file, encoding='utf-8')
#         else:
#             pass
#
# if __name__ == '__main__':
#     inputpath = './newimg/combine'  #此处替换为自己的路径
#     changelabelname(inputpath)


import os
import xml.etree.ElementTree as ET

#程序功能：批量修改VOC数据集中xml标签文件的标签名称
def changelabelname(inputpath):
    p = 0
    listdir = os.listdir(inputpath)
    for file in listdir:
        t = file
        if file.endswith('xml'):
            file = os.path.join(inputpath,file)
            tree = ET.parse(file)
            root = tree.getroot()
            for object1 in root.findall('filename'):
                object1.text = t.split('.')[0]+'.jpg'
                tree.write(file, encoding='utf-8')
            p += 1
        else:
            pass

if __name__ == '__main__':
    inputpath = './newimg/combine'  #此处替换为自己的路径
    changelabelname(inputpath)

# import os
# import xml.etree.ElementTree as ET
#
#
# def changelabelname(inputpath):
#     for items1 in os.listdir(inputpath):
#         path = inputpath + items1 + '/'
#         for items2 in os.listdir(path):
#             path1 = path + items2 + '/' + 'Annotation' + '/' + items2 + '/'
#             listdir = os.listdir(path1)
#             i =0
#             for file in listdir:
#                 i += 1
#                 if file.endswith('xml'):
#                     file = os.path.join(path1, file)
#                     tree = ET.parse(file)
#                     root = tree.getroot()
#                     for object1 in root.findall('object'):
#                         for sku in object1.findall('name'):           #查找需要修改的名称
#                             if (sku.text == items2):               #‘preName’为修改前的名称
#                                 sku.text = items1                 #‘TESTNAME’为修改后的名称
#                                 tree.write(file, encoding='utf-8')     #写进原始的xml文件并避免原始xml中文字符乱码
#                             else:
#                                 pass
#                 else:
#                     pass
#             print(items1, i)
#
# if __name__ == '__main__':
#     inputpath = './imgnet/flower/'
#     changelabelname(inputpath)