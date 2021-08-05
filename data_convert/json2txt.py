import os, json
import os.path as osp
import argparse


def json2csv(json_path, test_json, format='mmdetection'):
    "水下目标检测比赛里，将mmdetection的json输出转成比赛要求的csv格式数据"

    "yolov5输出的json文件里类名是从0开始算的，而mmdet是从1开始算的，不过mmdet的直接输出也是从0算，只不过转json的时候处理了一下"
    if format == "mmdetection":
        category = {1: "Airport", 2: "Port"}
    elif format == 'yolo':
        category = {0: "echinus", 1: "scallop", 2: "starfish", 3: "holothurian", 4: "waterweeds"}

    id2_filename = {}
    with open(test_json, "r", encoding='utf-8') as tf:
        ts_js = json.load(tf)

    for _recd in ts_js['images']:
        id2_filename[_recd['id']] = _recd['file_name'].strip('.tif')  #

    out_dir = os.path.dirname(json_path)
    out_dir = os.path.join(out_dir, "detection-results")
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r", encoding='utf-8') as f:
        jsRecord = json.load(f)

    result_image_dict = {}
    for one_ob in jsRecord:

        one_ob['bbox'].append(one_ob['score'])
        one_ob['bbox'].append(one_ob['category_id'])
        if one_ob['image_id'] not in result_image_dict:
            result_image_dict[one_ob['image_id']] = []

        result_image_dict[one_ob['image_id']].append(one_ob['bbox'])

    for img_ret in result_image_dict.keys():
        txt_name = "%s.txt" % id2_filename[img_ret]
        txt_path = os.path.join(out_dir, txt_name)

        with open(txt_path, 'w', encoding='utf-8') as txt:

            for obs in result_image_dict[img_ret]:
                name = category[obs[-1]]
                obs[2] += obs[0]
                obs[3] += obs[1]
                obs.pop(-1)
                obs.append(name)
                txt.write(
                    "%s  %f  %d %d %d %d" % (obs[-1], obs[-2], int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])))
                txt.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',
                        default='/home/data1/yw/data/iobjectspy_out/mmdetection/history_test_result/xf_result/xf_600_300.json.bbox.json')
    parser.add_argument('--test_json_path',
                        default='/home/data1/yw/data/mmdetection_data/airport_port_det_kdxf/k-fold/fold_v1/test/test.json')
    parser.add_argument('--json_format', default="mmdetection")

    arg = parser.parse_args()

    json2csv(arg.json_path, arg.test_json_path, arg.json_format)
