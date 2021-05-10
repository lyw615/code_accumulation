import os, json
import os.path as osp
from PIL import Image, ImageDraw


def show_two_image(image1, image2):
    # 同时可视化两个RGB或者mask
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    plt.imshow(image1)
    plt.sca(ax2)
    plt.imshow(image2)
    plt.show()


def json2csv():
    json_path = r"/home/data1/yw/data/iobjectspy_out/mmdetection/cascade_rcnn_lr_epoch/epoch35_out.bbox.json"
    js_base = osp.basename(json_path).strip(".json")
    csv_path = osp.join(osp.dirname(json_path), js_base.replace(".", "_") + ".csv")

    with open(json_path, "r", encoding='utf-8') as f:
        jsRecord = json.load(f)

    if osp.exists(csv_path):
        os.remove(csv_path)
        # raise  ValueError

    columns = ["name", "image_id", "confidence", "xmin", "ymin", "xmax", "ymax"]
    category = {1: "echinus", 2: "scallop", 3: "starfish", 4: "holothurian"}

    image_dir = r"/home/data1/yw/water_detection/test-A-image"
    aa = [x['image_id'] == 0 for x in jsRecord]
    import numpy as np
    inds = np.where(aa == True)[0]
    bb = aa[inds]
    all_content = []
    for line in jsRecord[:5]:
        wlines = []
        name = category[line['category_id']]

        image_id = "%06d" % (int(line['image_id']) + 1)
        bbox = line['bbox']

        bbox = list(map(int, bbox))
        # bbox[2]=bbox[0]+bbox[2]
        # bbox[3]=bbox[1]+bbox[3]
        show_img = Image.open(osp.join(image_dir, image_id + ".jpg"))

        draw = ImageDraw.Draw(show_img)
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=3)
        import numpy as np
        mat_show = np.array(show_img)
        print(bbox)
        show_two_image(mat_show, mat_show)

        confidence = line['score']

        wlines.append(name)
        wlines.append(image_id)
        wlines.append(confidence)
        wlines.extend(bbox)

        all_content.append(wlines)

    # import pandas as pd
    # pd_data=pd.DataFrame(all_content,columns=columns)
    # pd_data.to_csv(csv_path,index=False)

    for result in all_content:
        image_name = result[1] + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        bbox = result[3:]

        show_img = Image.open(image_path)
        show_img.show()
        draw = ImageDraw.Draw(show_img)
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])])

        show_img.show()


if __name__ == "__main__":
    "little using"
    json2csv()
