import os,sys,cv2
import numpy as np
from tqdm import tqdm
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..","..", "..","..")))

from code_aculat.utils.xml_process import analyze_xml

image_dir=r"/home/data1/yw/data/compt_data/qzb_data/Casia_modify/train/Images"
xml_dir=r"/home/data1/yw/data/compt_data/qzb_data/Casia_modify/train/Annotations"
dd={"10":"10","11":"11","21":"21","22":"22","23":"23","15":"15","52":"52"}
dd={"11":"11","12":"12","13":"13","14":"14","15":"15","16":"16","17":"17","18":"18","19":"19","10":"10"}
out_dir=r"/home/data1/yw/data/compt_data/qzb_data/Casia_modify/image_cls"
suffix='.jpg'

xmls=os.listdir(xml_dir)
for xml in tqdm(xmls):
    xml_path=os.path.join(xml_dir,xml)
    class_names,rectangle=analyze_xml(xml_path,check_bbox=True)

    if len(class_names):
        image_name=xml.split('.')[0]+suffix
        image=cv2.imdecode(np.fromfile(os.path.join(image_dir,image_name),dtype=np.uint8),1)[:,:,::-1]
        height,width,_=image.shape

        for  i in range(len(class_names)):
            if class_names[i] in dd.keys():
                xmin,ymin,xmax,ymax=rectangle[i]
                # image_size=max(xmax-xmin,ymax-ymin,224)
                # image_size=int(max(xmax-xmin,ymax-ymin)*1.43)
                image_size=int(max(xmax-xmin,ymax-ymin))

                if xmax-xmin <image_size:
                    xmin-=(image_size-(xmax-xmin))//2
                    xmin=max(0,xmin)
                    xmax=min(xmin+image_size,width-1)

                if ymax-ymin <image_size:
                    ymin-=(image_size-(ymax-ymin))//2
                    ymin=max(0,ymin)
                    ymax=min(ymin+image_size,height-1)

                bbox_img=image[ymin:ymax,xmin:xmax,:]

                save_dir=os.path.join(out_dir,dd[class_names[i]])
                os.makedirs(save_dir,exist_ok=True)
                start_num=len(os.listdir(save_dir))
                save_path=os.path.join(save_dir,"%d.jpg"%start_num)
                if os.path.exists(save_path):
                    raise("file has existed  %s"%save_path)
                cv2.imencode(".jpg",bbox_img)[1].tofile(save_path)



