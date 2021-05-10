import os, json
import os.path as osp


def json2csv(json_path,format='mmdetection'):
    "format  json result from mmdetection or yolov"

    # json_path = r"/home/data1/yw/water_detection/split_folds/fold_v1/test.json"
    js_base = osp.basename(json_path).strip(".json")
    csv_path = osp.join(osp.dirname(json_path), js_base.replace(".", "_") + ".csv")

    with open(json_path, "r", encoding='utf-8') as f:
        jsRecord = json.load(f)

    #test waterweeds
    # for inst in jsRecord['annotations']:
    #     if inst['category_id']==5:
    #         print(inst)

    if osp.exists(csv_path):
        os.remove(csv_path)
        # raise  ValueError

    columns = ["name", "image_id", "confidence", "xmin", "ymin", "xmax", "ymax"]

    if format=="mmdetection":
        category = {1: "echinus", 2: "scallop", 3: "starfish", 4: "holothurian",5:"waterweeds"}
    elif format=='yolo':
        category = {0: "echinus", 1: "scallop", 2: "starfish", 3: "holothurian",4:"waterweeds"}


    all_content = []
    for line in jsRecord:
        wlines = []
        name = category[line['category_id']]

        if name=="waterweeds":
            continue

        image_id = "%06d" % int(line['image_id'])
        bbox = line['bbox']

        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox = list(map(int, bbox))

        confidence = line['score']

        wlines.append(name)
        wlines.append(image_id)
        wlines.append(confidence)
        wlines.extend(bbox)

        all_content.append(wlines)

    import pandas as pd
    pd_data = pd.DataFrame(all_content, columns=columns)
    pd_data.to_csv(csv_path, index=False)

    print("csv file saved in %s"%csv_path)


if __name__ == "__main__":
    "convert mmdetection json prediction to submit csv format"
    json_path = r"/home/data1/yw/data/iobjectspy_out/mmdetection/history_test_result/no_influence/pesual_lowLr_mAP47.bbox.json"
    json_path = r"/home/data1/yw/github_projects/yolov5/runs/test/exp/best_predictions.json"
    json_format="yolo"
    json2csv(json_path,json_format)
