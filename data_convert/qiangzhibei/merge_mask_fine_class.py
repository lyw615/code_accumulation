import os,json
import numpy as np
import sys,cv2
from tqdm import  tqdm


file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..","..", "..", "..")))
print(os.path.abspath(os.path.join(file_path, "..","..", "..", "..")))

from code_aculat.utils.xml_process  import analyze_xml
from code_aculat.data_operation.gt_pre_iou_analyse import bbox_iou
from code_aculat.visualize.visual_base import draw_bboxes_on_image
xml_dir=r"G:\first_select_Annotations"
mask_dir=r"G:\hrsc\mask"
image_dir=r"G:\hrsc\Images"

out_dir=r"G:\hrsc\fine_ship"
out_fig=r"G:\hrsc\out_fig"
os.makedirs(out_dir)

mask_json=os.listdir(mask_dir)

# for mask in tqdm(mask_json):
for mask in mask_json:
    xml_file=os.path.join(xml_dir,mask.replace(".json",'.xml'))
    json_file=os.path.join(mask_dir,mask)
    class_name, rectangle_position=analyze_xml(xml_file)

    if len(rectangle_position)<1:
        print('empty xml  %s'%xml_file)
        continue

    rectangle_position=np.array(rectangle_position)


    with open(json_file) as js:
        mask_js=json.load(js)

    delete_mask_inds=[]
    for _  in range(len(mask_js['shapes'])):
        shape=mask_js['shapes'][_]
        mask_bboxes = []
        points = np.array(shape['points'], dtype=np.float32)
        # xx= cv2.minAreaRect(points)
        x, y, w, h = cv2.boundingRect(points)
        mask_bboxes.append([x,y,x+w,y+h])

        #cal iou with other bboxes
        mask_bboxes=np.array(mask_bboxes,dtype=np.int)
        ious=bbox_iou(mask_bboxes,rectangle_position)
        iou=ious[0]
        try:
            match_ind=np.argmax(iou)
        except:
            pass

        if iou[match_ind]>0.7:
            mask_js['shapes'][_]['label']=class_name[match_ind]

            rectangle_position=np.delete(rectangle_position,match_ind,axis=0)  #匹配成功，删除xml里面的bbox
            class_name.pop(match_ind)

            if len(rectangle_position)<1:   #说明此时xml里的对象已经匹配完了
                break
        else:
            print(iou[match_ind])
            delete_mask_inds.append(mask_js['shapes'][_])   #匹配失败，删除json里多余的mask

    for i_shape in delete_mask_inds:
        mask_js['shapes'].remove(i_shape)

    if len(rectangle_position)>0:
        image_path=os.path.join(image_dir,mask.replace(".json",'.bmp'))
        draw_bboxes_on_image(image_path,rectangle_position,class_name,save_dir=out_fig,save_name=os.path.basename(image_path).split(".")[0]+'.png')
        # print("there are %d label not marked as mask  in file  %s "%(len(rectangle_position),os.path.basename(xml_file)),rectangle_position,class_name)

    if len(mask_js['shapes'])>0:

        # 保存json文件
        json.dump(mask_js, open(os.path.join(out_dir, mask), 'w'), indent=4)