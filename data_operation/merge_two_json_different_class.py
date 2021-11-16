import json
import os

jf_base = json.load(
    open("/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv/Json/raw_transpose.json", "r"))  # bigger json file
jf_add = json.load(open("/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv/104_tv39_hrsc.json", "r"))

out_json_path = "/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv/104_tv39_hrsc_raw_trans.json"

# process image merge
base_images = jf_base['images']
add_images = jf_add['images']
new_js_dict = {}
start_img_id = len(base_images)
start_obj_id = len(jf_base['annotations'])

for image_dic in add_images:
    image_dic['id'] = image_dic['id'] + start_img_id
    base_images.append(image_dic)

# process  class merge
# add_old_catid2new_dic = {
#     21: 12, 22: 12, 23: 12,
#     1: 10, 2: 11, 15: 15
# }
add_old_catid2new_dic = {
    10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19
}

base_old_catid2new_dic = {
    10: 10, 11: 11, 21: 12, 22: 12, 16: 16, 31: 13, 32: 13, 41: 14, 51: 15, 52: 15, 71: 17, 72: 17, 81: 18, 82: 18,
    91: 19, 92: 19
}

# base_old_catid2new_dic = {
#     10: 10, 11: 11, 12: 12,  13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19
# }


new_categories_dic = {"10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18,
                      "19": 19}

new_categories = []
for key, value in new_categories_dic.items():
    new_categories.append({'id': value, 'name': key})

# process objs merge
base_annos = jf_base["annotations"]
add_annos = jf_add["annotations"]

for anno in base_annos:
    anno['category_id'] = base_old_catid2new_dic[anno['category_id']]

for anno in add_annos:
    anno['category_id'] = add_old_catid2new_dic[anno['category_id']]
    anno['id'] += start_obj_id
    anno['image_id'] += start_img_id
    base_annos.append(anno)

# new_js_dict['type']=jf_add["type"]
new_js_dict['images'] = base_images
new_js_dict['annotations'] = base_annos
new_js_dict['categories'] = new_categories

json.dump(new_js_dict, open(out_json_path, 'w'))
