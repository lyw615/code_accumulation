import os
import shutil

import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import codecs,xml

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


annotations_dir=r"/home/data1/GeoAI_Data/warship_detect_HRSC/HRSC2016_part1/Train/Annotations"
annotations_dir=r"/home/data1/GeoAI_Data/warship_detect_HRSC/HRSC2016_part1/FullDataSet/Annotations"
outdir=os.path.dirname(annotations_dir)
outdir=os.path.join(outdir,"first_select_Annotations")
os.makedirs(outdir,exist_ok=True)



# PRE_DEFINE_CATEGORIES = {"1": 1,"2": 2,"15": 15,"17": 17, "21": 21,"22": 22,"23": 23,}
# #将类别初步做映射,先使得长尾分布不那么严重,方便选项目
class_dict={'100000002':1,
'100000005':1,
'100000006':1,
'100000012':1,
'100000013':1,
'100000032':1,

'100000007':21,
'100000009':22,
'100000014':22,
'100000011':23,

'100000008':15,

'100000010':17,
'100000015':17,

'100000016':2}

xmls=os.listdir(annotations_dir)
xmls=list(filter(lambda x:x.endswith('.xml'),xmls))
low_res_num=0
error_bb=0

for xml in tqdm(xmls):
    xml_path=os.path.join(annotations_dir,xml)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    path = get(root, "Img_FileName")
    if len(path) == 1:
        filename = os.path.basename(path[0].text)
    else:
        raise ValueError("%d paths found in %s" % (len(path), xml_path))
    ## The filename must be a number

    try:
        resolution = float(get_and_check(root, "Img_Resolution", 1).text)
    except:
        resolution=float(get_and_check(root, "Img_Resolution", 1).text.split('(')[0])


    place_id = get_and_check(root, "Place_ID", 1).text
    width = int(get_and_check(root, "Img_SizeWidth", 1).text)
    height = int(get_and_check(root, "Img_SizeHeight", 1).text)

    img_suffix = get_and_check(root, "Img_FileFmt", 1).text
    filename="%s.%s"%(filename,img_suffix)

    if resolution>1.1:
        low_res_num+=1
        continue


    saved_path=os.path.join(outdir,xml)
    channels=3

    bbox_error=True
    # xml 头文件信息
    with codecs.open(saved_path, "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC' + '</folder>\n')
        xml.write('\t<filename>' + filename + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t\t<resolution>' + str(resolution) + '</resolution>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')

        HRSC_Objects=get(root, "HRSC_Objects")[0]
        for obj in get(HRSC_Objects, "HRSC_Object"):
            category_id = get_and_check(obj, "Class_ID", 1).text

            if category_id not  in class_dict.keys():
                continue

            new_cate_id=str(class_dict[category_id])

            xmin = int(get_and_check(obj, "box_xmin", 1).text)
            ymin = int(get_and_check(obj, "box_ymin", 1).text)
            xmax = int(get_and_check(obj, "box_xmax", 1).text)
            ymax = int(get_and_check(obj, "box_ymax", 1).text)

            if ymin>=ymax or xmin>=xmax:
                continue

            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + new_cate_id + '</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>0</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')

            bbox_error=False

        xml.write('</annotation>')
    if bbox_error:
        os.remove(saved_path)
        error_bb+=1

print(low_res_num)
print(error_bb)