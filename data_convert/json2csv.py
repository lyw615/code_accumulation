import os, json
import os.path as osp
import argparse


def json2csv(json_path, format='mmdetection'):
    "水下目标检测比赛里，将mmdetection的json输出转成比赛要求的csv格式数据"

    "csv 文件的列名"
    columns = ["name", "image_id", "confidence", "xmin", "ymin", "xmax", "ymax"]

    "yolov5输出的json文件里类名是从0开始算的，而mmdet是从1开始算的，不过mmdet的直接输出也是从0算，只不过转json的时候处理了一下"
    if format == "mmdetection":
        category = {1: "echinus", 2: "scallop", 3: "starfish", 4: "holothurian", 5: "waterweeds"}
    elif format == 'yolo':
        category = {0: "echinus", 1: "scallop", 2: "starfish", 3: "holothurian", 4: "waterweeds"}

    js_base = osp.basename(json_path).strip(".json")
    csv_path = osp.join(osp.dirname(json_path), js_base.replace(".", "_") + ".csv")

    with open(json_path, "r", encoding='utf-8') as f:
        jsRecord = json.load(f)

    if osp.exists(csv_path):
        os.remove(csv_path)
        print('the exist csv file has been deleted')

    all_content = []
    for line in jsRecord:
        wlines = []
        name = category[line['category_id']]

        if name == "waterweeds":
            "这个类别不计入提交结果"
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

    print("csv file saved in %s" % csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path')
    parser.add_argument('--json_format', default="yolo")

    arg = parser.parse_args()

    json2csv(arg.json_path, arg.json_format)
