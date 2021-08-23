import argparse
import shutil
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./data/", type=str)
    args = parser.parse_args()
    data_path = args.data_path

    if not data_path.endswith('/'):
        data_path = data_path+'/'
    if os.path.exists('./images'):
        shutil.rmtree('./images')
    os.mkdir('./images')
    os.mkdir('./images/train')
    os.mkdir('./images/test')

    
    for i in ['train', 'test']:
        f = open(data_path+'ImageSets/Main/'+i+'.txt',"r")
        lines = f.readlines()
        for file_name in lines:
            img = file_name.rstrip('\r\n')+'.jpg'
            xml = file_name.rstrip('\r\n')+'.xml'
            shutil.copy(data_path+'JPEGImages/'+img, './images/'+i)
            shutil.copy(data_path+'Annotations/'+xml, './images/'+i)
        
        image_path = os.path.join(os.getcwd(), ('images/' + i))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/' + i + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')
