import os,json
label_js_path=r"/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold-v1/fold_v1/new1_train.json"
label_js=json.load(open(label_js_path,'r'))

images=label_js['images']
changed=False
for anno in label_js["annotations"]:
        image_width=images[anno['image_id']-1]['width']
        image_height=images[anno['image_id']-1]['height']

        if anno['bbox'][0]+anno['bbox'][2]>=image_width or anno['bbox'][1]+anno['bbox'][3]>=image_height:
            label_js["annotations"].remove(anno)
            changed=True

if changed:
    save_path=os.path.join(os.path.dirname(label_js_path),"new_%s"%(os.path.basename(label_js_path)))
    with open(save_path,'w') as f:
        json.dump(label_js,f)
    print("new json saved in %s"%save_path)