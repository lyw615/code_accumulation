import sys, os

file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..")))

from code_aculat.data_analyse.data_analyse import analyse_image_wh, output_big_wh
from code_aculat.data_operation.crop_voc_2small_tile import slice_im

size_thresh=10000   #边长达到多少，即认为是需要被裁剪的图片
xml_names=output_big_wh(r"/home/data1/GeoAI_Data/compete_unzip/train/Annotations",size_thresh)


raw_images_dir = r'E:\push_branch\resources_ml\out\training\object_ext_train_data\Images'  # 这里就是原始的图片
raw_ann_dir = r'E:\push_branch\resources_ml\out\training\object_ext_train_data\Annotations'

output_dir = r"E:\push_branch\resources_ml\out\clip_xml"
out_voc_dir = os.path.join(output_dir, "Annotations")  # 切出来的标签也保存为voc格式
out_image_dir = os.path.join(output_dir, "Images")
os.makedirs(out_voc_dir, exist_ok=True)
os.makedirs(out_image_dir, exist_ok=True)

sliceHeight = 128
sliceWidth = 128
overlap = 30
area_thresh = 0.7

List_imgs = [x.replace(".xml",".tif")  for x in xml_names]  # .tif 表示图片的后缀


slice_im(List_imgs, out_image_dir, out_voc_dir, raw_images_dir, raw_ann_dir, sliceWidth=sliceWidth,
             sliceHeight=sliceHeight,
             overlap=overlap, area_thresh=area_thresh)  # 切片的宽高,切片的重叠，label的面积保留比例