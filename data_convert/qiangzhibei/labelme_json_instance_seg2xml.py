import json, codecs, os
from tqdm import tqdm
import numpy as np

# ±êÇ©Â·¾¶
labelme_path = "/home/data1/competition/data/tianzhi2/CASIA-Ship/train/mask"  # Ô­Ê¼labelme±ê×¢Êý¾ÝÂ·¾¶
saved_path = "/home/data1/competition/data/tianzhi2/CASIA-Ship/train/Annotations"  # ±£´æÂ·¾¶
os.makedirs(saved_path, exist_ok=True)

# »ñÈ¡´ý´¦ÀíÎÄ¼þ
files = os.listdir(labelme_path)
files = list(filter(lambda x: x.endswith('json'), files))
files = [i.split("/")[-1].split(".json")[0] for i in files]

# ¶ÁÈ¡±ê×¢ÐÅÏ¢²¢Ð´Èë xml
for json_file_ in tqdm(files):
    json_filename = os.path.join(labelme_path, json_file_ + ".json")
    json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    height, width, channels = json_file['imageHeight'], json_file['imageWidth'], 3
    with codecs.open(os.path.join(saved_path, json_file_ + ".xml"), "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + label + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>0</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(int(round(xmin))) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(int(round(ymin))) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(int(round(xmax))) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(int(round(ymax))) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')

        xml.write('</annotation>')
