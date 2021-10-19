import sys,os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "..", "..","..", "..")))
from code_aculat.visualize.visual_base import draw_multi_bboxes
import numpy as np
import json,copy
import cv2 as cv
from PIL import  Image,ImageDraw
import random
import matplotlib.pyplot as plt
from  albumentations import Resize,HorizontalFlip,VerticalFlip,ChannelShuffle
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from pycocotools.coco import COCO

def rand(a=0, b=1):
    # 总区间的比例再加上区间下限，生成随机数
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty, h, w, save_proportion):
    """
    对于合成到新图像中的bbox，单个部分超出cutx,y以及新图片宽高的进行删除或者修改，保留不低于原来面积指定比例的bbox
    """
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            raw_x1, raw_y1, raw_x2, raw_y2 = box[0], box[1], box[2], box[3]
            clipped = False

            if i == 0:
                # 合成后的图被cutx,cuty分成四个部分，单个图超出界限的将被覆盖，这里y1,x1就是判断是否超出该部分界限，以下i==之后的第一个判断都是这个意图
                if y1 > cuty or x1 > cutx:
                    continue
                # ***处理x和y方向上，跨越界限的y2或者x2,如果这个bbox这样处理之后在x,y任一方向上小于5个像素，那就舍弃改bbox,而我也是在这里判断保留比例
                if y2 >= cuty and y1 <= cuty:
                    clipped = True
                    y2 = cuty

                if x2 >= cutx and x1 <= cutx:
                    clipped = True
                    x2 = cutx

                if clipped:
                    cut_area = (x2 - x1) * (y2 - y1)
                    raw_area = (raw_x2 - raw_x1) * (raw_y2 - raw_y1)
                    if not (cut_area / raw_area >= save_proportion):
                        continue

            elif i == 1:
                if y2 < cuty or x1 > cutx or y1 > h:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    clipped = True
                    y1 = cuty

                if x2 >= cutx and x1 <= cutx:
                    clipped = True
                    x2 = cutx

                if clipped:
                    cut_area = (x2 - x1) * (y2 - y1)
                    raw_area = (raw_x2 - raw_x1) * (raw_y2 - raw_y1)
                    if not (cut_area / raw_area >= save_proportion):
                        continue

            elif i == 2:
                if y2 < cuty or x2 < cutx or x1 > w or y1 > h:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    clipped = True
                    y1 = cuty

                if x2 >= cutx and x1 <= cutx:
                    clipped = True
                    x1 = cutx

                if clipped:
                    cut_area = (x2 - x1) * (y2 - y1)
                    raw_area = (raw_x2 - raw_x1) * (raw_y2 - raw_y1)
                    if not (cut_area / raw_area >= save_proportion):
                        continue

            elif i == 3:
                if y1 > cuty or x2 < cutx or x1 > w:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    clipped = True
                    y2 = cuty

                if x2 >= cutx and x1 <= cutx:
                    clipped = True
                    x1 = cutx

                if clipped:
                    cut_area = (x2 - x1) * (y2 - y1)
                    raw_area = (raw_x2 - raw_x1) * (raw_y2 - raw_y1)
                    if not (cut_area / raw_area >= save_proportion):
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])  # bbox有5列,最后一列是类别编号
            merge_bbox.append(tmp_box)
    return merge_bbox

def show_two_image(image1, image2, title=None):
    # 同时可视化两个RGB或者mask
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    plt.imshow(image1)
    plt.sca(ax2)
    plt.imshow(image2)
    if title:
        plt.title(title)
    plt.show()

def get_rgb(v):
    """
    获取RGB颜色
   :param v: 十六进制颜色码
   :return: RGB颜色值
       """
    r, g, b = v[1:3], v[3:5], v[5:7]
    return int(r, 16), int(g, 16), int(b, 16)

# def get_resize_scale(index,):

def get_mosaic_coco(save_proportion=0.9,
                    hue=.1, sat=1.5, val=1.5, proc_img=True, color_p=(3, 6), color_aug=False):

    json_path=r"G:\hrsc\annotations\train.json"
    image_dir=r"G:\hrsc\images\val"

    coco = COCO(json_path)
    imgIds=coco.getImgIds()
    catIds=coco.getCatIds()
    seleted_imgids = np.random.choice(imgIds, 4)

    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    mask_datas = []
    bbox_datas = []
    index = 0
    # place_x,y组合起来就是这张图复制到新图上，所在的左上角坐标

    # images=jf['images']
    # imgid2img_info={}
    # imgid2annos={}
    # for anno in jf['annotations']:
    #     if anno['image_id'] not in imgid2annos.keys():
    #         imgid2annos[anno['image_id']]=[]
    #
    #     imgid2annos[anno['image_id']].append(anno)
    #
    # for img in jf['images']:
    #     imgid2img_info[img['id']]=img
    h,w=0,0
    for imid in seleted_imgids: #选中图片里最大的作为最终合成图片的size
        img=coco.loadImgs(int(imid))
        if not img:
            break

        img=img[0]
        h=max(img['height'],h)
        w=max(img['width'],w)

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]

    #处理单张数据
    for imid in seleted_imgids:

        #获取该图上的mask
        img=coco.loadImgs(int(imid))
        if not img:
            break

        img=img[0]

        annIds=coco.getAnnIds(imgIds=img['id'],catIds=catIds,iscrowd=None)
        annos=coco.loadAnns(annIds)

        # 获取该图上的bbox和class_id
        cat_ids_img=[]
        bboxes_img=[]

        if len(annos)>0:
            mask_raw=coco.annToMask(annos[0])*annos[0]['category_id']
            cat_ids_img.append(annos[0]['category_id'])
            bboxes_img.append(annos[0]['bbox'])
            for i in range(len(annos)-1):
                mask_raw+=coco.annToMask(annos[i+1])*annos[i+1]['category_id']
                cat_ids_img.append(annos[i+1]['category_id'])
                bboxes_img.append(annos[i+1]['bbox'])

            img_origin_path=os.path.join(image_dir,img['file_name'])
            # 打开图片
            image = Image.open(img_origin_path)
            image = image.convert("RGB")

            # 打开图片的大小
            iw, ih = image.size

            if len(bboxes_img) < 1:
                "没有标签,那后续的操作无法进行"
                break

            bbox = np.array(bboxes_img)
            bbox[:,2]+=bbox[:,0]
            bbox[:,3]+=bbox[:,1]

            # shw_img = image
            # draw_multi_bboxes(shw_img, bbox)
            # drawed_img = np.array(shw_img, dtype=np.uint8)
            # show_two_image(drawed_img, drawed_img)

            bbox = np.insert(bbox, 4, cat_ids_img, axis=1)

            # image.save(str(index)+".jpg")
            # 是否翻转图片
            flip = rand() < .5
            if flip and len(bbox) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_raw=cv.flip(mask_raw, flipCode=1)
                bbox[:, [0, 2]] = iw - bbox[:, [2, 0]]

            # 将图片进行放置，分别对应四张分割图片的左上角位置
            dx = place_x[index]
            dy = place_y[index]

            # 缩放后的图片应该能够填充满指定区域
            if index == 0:
                min_x, min_y = place_x[2], place_y[2]
            elif index == 1:
                min_x, min_y = place_x[2], h - place_y[2]
            elif index == 2:
                min_x, min_y = w - place_x[2], h - place_y[2]
            elif index == 3:
                min_x, min_y = w - place_x[2], place_y[2]


            scale = rand(scale_low, scale_high)

            nw,nh=int(iw*scale),int(ih*scale)
            min_scale=min(nw/min_x,nh/min_y,1)+0.001
            nw,nh=int(nw/min_scale),int(nh/min_scale)

            # show_two_image(image,mask_raw)
            # 图片resize到指定尺寸
            image = image.resize((nw, nh), Image.BICUBIC)
            mask = cv.resize(mask_raw, dsize=(nw, nh), interpolation=cv.INTER_NEAREST)

            # show_two_image(image,mask)
            # 进行色域变换，用现成的
            hue = rand(-hue, hue)
            sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
            val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
            x = rgb_to_hsv(np.array(image) / 255.)
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            image = hsv_to_rgb(x)

            image = Image.fromarray((image * 255).astype(np.uint8))
            # **image.show()

            # w h 是设定的合成后图片的宽高
            new_image = Image.new('RGB', (w, h))
            new_mask = np.zeros(shape=(h,w),dtype=np.uint8)
            new_mask=Image.fromarray(new_mask)
            # dx, dy是左上角的坐标点,paste会有损失吗？有的，只指定了左上角，溢出的就浪费了
            new_image.paste(image, (dx, dy))
            mask = Image.fromarray(mask)
            new_mask.paste(mask, (dx, dy))

            #用mask去填new mask那点右下部分，填不满就算了，超过就取部分
            # new_mask[dx:,dy:]=mask[:w-dx,:h-dy]
            image_data = np.array(new_image)

            if color_aug:
                "在拼图中随机使用颜色增强"
                apply_p = random.randint(color_p[0], color_p[1]) / 10
                if random.randint(0, 10) / 10 > apply_p:
                    color_change = random.randint(0, 1)
                    if color_change == 0:
                        image_data = ChannelShuffle(p=1)(image=image_data)["image"]
                    elif color_change == 1:
                        image_gray = cv.cvtColor(image_data, cv.COLOR_RGB2GRAY)
                        image_data = cv.cvtColor(image_gray, cv.COLOR_GRAY2RGB)

            mask_data = np.array(new_mask,dtype=np.uint8)
            # show_two_image(image_data,mask_data)
            image_data = image_data / 255

            index = index + 1
            bbox_data = []
            # 对bbox进行重新处理
            if len(bbox) > 0:
                # 根据新的宽高与原来宽高的比例得到缩放后的坐标，再加上这张图片在新的大图上的左上角坐标
                bbox[:, [0, 2]] = bbox[:, [0, 2]] * nw / iw + dx
                bbox[:, [1, 3]] = bbox[:, [1, 3]] * nh / ih + dy

                raw_bbox = copy.deepcopy(bbox)

                # 小于0的都变成0，超过w,h的bbox值都变成w,h
                bbox[:, 0:2][bbox[:, 0:2] < 0] = 0

                ##超过像素索引,255之后会从0开始,-1已经到像素的最大索引了
                bbox[:, 2][bbox[:, 2] > (w - 1)] = w - 1
                bbox[:, 3][bbox[:, 3] > (h - 1)] = h - 1

                # 计算前后的面积比例,保留大于一定比例的标注
                now_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
                raw_areas = (raw_bbox[:, 2] - raw_bbox[:, 0]) * (raw_bbox[:, 3] - raw_bbox[:, 1])

                save_index = np.where(now_areas / raw_areas > save_proportion)[0]
                bbox = bbox[save_index]

                # #可视化缩放后的bbox是否能重叠到贴上去的图
                # shw_img = Image.fromarray(np.array(image_data *255,dtype=np.uint8))
                # draw_multi_bboxes(shw_img, bbox)
                # drawed_img = np.array(shw_img, dtype=np.uint8)
                # show_two_image(drawed_img, new_mask)

                # ** "用面积过滤指定类别对象"
                # both_save_inds=np.array([])
                # bask_inds=np.where(bbox[:,4]==bask_num)[0]
                # bask_save_inds=np.where(now_areas[bask_inds]>3200)[0]
                # both_save_inds=both_save_inds+bask_inds[bask_save_inds]
                #
                # trace_inds = np.where(bbox[:, 4] == trace_num)[0]
                # trace_save_inds = np.where(now_areas[trace_inds] > 43000)[0]
                # both_save_inds = both_save_inds + trace_inds[trace_save_inds]
                #
                # bbox=bbox[both_save_inds]

                if len(bbox) > 0:
                    bbox_w = bbox[:, 2] - bbox[:, 0]
                    bbox_h = bbox[:, 3] - bbox[:, 1]
                    # 需要bbox的宽高都大于1的bbox才能进入下一步，and表示为与操作，两者都正确则结果正确,bbox[True,True,True...]
                    bbox = bbox[np.logical_and(bbox_w > 1, bbox_h > 1)]

                    bbox_data = np.zeros((len(bbox), 5))
                    bbox_data[:len(bbox)] = bbox


            image_datas.append(image_data)
            mask_datas.append(mask_data)
            bbox_datas.append(bbox_data)

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    new_image = (new_image * 255).astype(np.uint8)

    # **#检验第四个顺序的图bbox对应标签
    # mmask=Image.fromarray((mask_datas[3]*255).astype(np.uint8))
    # for num in range(len(bbox_datas[3])):
    #     left, top, right, bottom = bbox_datas[3][num]
    #     draw = ImageDraw.Draw(mmask)
    #     draw.rectangle([left , top , right, bottom ], outline="green",width=2)
    #
    # mmask.show()

    new_mask = np.zeros([h, w])
    new_mask[:cuty, :cutx] = mask_datas[0][:cuty, :cutx]
    new_mask[cuty:, :cutx] = mask_datas[1][cuty:, :cutx]
    new_mask[cuty:, cutx:] = mask_datas[2][cuty:, cutx:]
    new_mask[:cuty, cutx:] = mask_datas[3][:cuty, cutx:]
    new_mask = new_mask.astype(np.uint8)

    # shw_img = Image.fromarray(new_image)
    # for i in range(len(bbox_datas)):
    #     draw_multi_bboxes(shw_img, bbox_datas[i])
    #     drawed_img = np.array(shw_img, dtype=np.uint8)
    #     show_two_image(drawed_img, new_mask)

    new_bbox = merge_bboxes(bbox_datas, cutx, cuty, h, w, save_proportion=save_proportion)
    # if len(new_bbox)>0:
    #     **"合并bbox后,再次用面积过滤指定类别对象"
    #     new_bbox=np.array(new_bbox)
    #     now_areas = (new_bbox[:, 2] - new_bbox[:, 0]) * (new_bbox[:, 3] - new_bbox[:, 1])
    #


    if len(new_bbox) > 0:
        new_bbox = np.array(new_bbox).astype(np.int)
        # [xmin:xmax,ymin:ymax]没取到xmax,ymax
        new_bbox[:, 2] = new_bbox[:, 2] + 1
        new_bbox[:, 3] = new_bbox[:, 3] + 1

        new_bbox[:, 2][new_bbox[:, 2] > w - 1] = w - 1
        new_bbox[:, 3][new_bbox[:, 3] > h - 1] = h - 1


        clean_mask=np.zeros(shape=new_mask.shape,dtype=np.uint8)

        masks=[]
        for  idx in range(len(new_bbox)):
            rectangle=new_bbox[idx]
            xmin, ymin, xmax, ymax,cls_num = rectangle[0], rectangle[1], rectangle[2], rectangle[3],rectangle[4]
            mask = np.zeros_like(new_mask, np.uint8)
            mask[ymin:ymax, xmin:xmax] = new_mask[ymin:ymax, xmin:xmax]

            # 计算矩形中点像素值
            mean_x = (xmin + xmax) // 2
            mean_y = (ymin + ymax) // 2

            end = min((mask.shape[1], round(xmax) + 1))
            start = max((0, round(xmin) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = new_mask[y_, x_]
                if pixels != 0:  # 0 对应背景
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = new_mask[y_, x_]
                    if pixels != 0:
                        mask = (mask == pixels).astype(np.uint8)
                        break

            masks.append(mask)
            clean_mask+=mask*cls_num

        shw_img=Image.fromarray(new_image)
        draw_multi_bboxes(shw_img,new_bbox)
        shw_mask = Image.fromarray(clean_mask)
        draw_multi_bboxes(shw_mask, new_bbox)
        drawed_img=np.array(shw_img,dtype=np.uint8)
        drawed_mask=np.array(shw_mask,dtype=np.uint8)
        show_two_image(drawed_img,drawed_mask)


        return new_image, new_mask, new_bbox
    else:
        return [], [], []
for i in range(11):
    get_mosaic_coco()

# def get_mosaic_data(images_dir,  files, class_dict, input_shape, save_proportion=0.9,
#                     hue=.1, sat=1.5, val=1.5, proc_img=True, color_p=(3, 6), color_aug=False):
#     '''random preprocessing for real-time data augmentation'''
#     h, w = input_shape
#     min_offset_x = 0.4
#     min_offset_y = 0.4
#     scale_low = 1 - min(min_offset_x, min_offset_y)
#     scale_high = scale_low + 0.2
#
#     image_datas = []
#     mask_datas = []
#     bbox_datas = []
#     index = 0
#     # place_x,y组合起来就是这张图复制到新图上，所在的左上角坐标
#     place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
#     place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
#     img_suffix = os.listdir(images_dir)
#     img_suffix = img_suffix[-1].split(".")[-1]
#     for mask_path in files:
#         # 每一行进行分割，一行对应一个图片的名称和图片里的bboxes
#
#         img_file = mask_file.replace("png", img_suffix)
#         img_path = os.path.join(images_dir, img_file)
#
#         xml_file = mask_file.replace("png", "xml")
#         xml_path = os.path.join(xmls_dir, xml_file)
#         # 打开图片
#         image = Image.open(img_path)
#         image = image.convert("RGB")
#         # 打开图片的大小
#         iw, ih = image.size
#         mask_raw =
#
#         # 解析出的bbox与要求的一致
#         class_name, bbox =
#         if len(bbox) < 1:
#             "没有标签,那后续的操作无法进行"
#             return [], [], []
#         class_id = np.array([class_dict[x] for x in class_name])
#         bbox = np.array(bbox)
#         bbox = np.insert(bbox, 4, class_id, axis=1)
#
#         # image.save(str(index)+".jpg")
#         # 是否翻转图片
#         flip = rand() < .5
#         if flip and len(bbox) > 0:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             cv.flip(mask_raw, flipCode=1, dst=mask_raw)
#             bbox[:, [0, 2]] = iw - bbox[:, [2, 0]]
#
#         # 对输入进来的图片进行缩放，缩放的比例跟合并后的图像宽高相关，与图片本身的宽高无关
#         new_ar = w / h
#         scale = rand(scale_low, scale_high)
#         # 对长度更多的那一边先缩放，在利用宽高比例来得到短边
#         if new_ar < 1:
#             nh = int(scale * h)
#             # 保持宽高比例，用缩放后的高求宽
#             nw = int(nh * new_ar)
#         else:
#             nw = int(scale * w)
#             nh = int(nw / new_ar)
#         # 图片resize到指定尺寸
#         image = image.resize((nw, nh), Image.BICUBIC)
#         mask = cv.resize(mask_raw, dsize=(nh, nw), interpolation=cv.INTER_NEAREST)
#
#         # 进行色域变换，用现成的
#         hue = rand(-hue, hue)
#         sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
#         val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
#         x = rgb_to_hsv(np.array(image) / 255.)
#         x[..., 0] += hue
#         x[..., 0][x[..., 0] > 1] -= 1
#         x[..., 0][x[..., 0] < 0] += 1
#         x[..., 1] *= sat
#         x[..., 2] *= val
#         x[x > 1] = 1
#         x[x < 0] = 0
#         image = hsv_to_rgb(x)
#
#         image = Image.fromarray((image * 255).astype(np.uint8))
#         # **image.show()
#         # 将图片进行放置，分别对应四张分割图片的位置
#         dx = place_x[index]
#         dy = place_y[index]
#         # w h 是设定的合成后图片的宽高，128设定的是color
#         new_image = Image.new('RGB', (w, h), (128, 128, 128))
#         new_mask = Image.new('RGB', (w, h), (128, 128, 128))
#         # dx, dy是左上角的坐标点,paste会有损失吗？有的，只指定了左上角，溢出的就浪费了
#         new_image.paste(image, (dx, dy))
#         mask = Image.fromarray(mask)
#         new_mask.paste(mask, (dx, dy))
#         image_data = np.array(new_image)
#
#         if color_aug:
#             "在拼图中随机使用颜色增强"
#             apply_p = random.randint(color_p[0], color_p[1]) / 10
#             if random.randint(0, 10) / 10 > apply_p:
#                 color_change = random.randint(0, 1)
#                 if color_change == 0:
#                     image_data = ChannelShuffle(p=1)(image=image_data)["image"]
#                 elif color_change == 1:
#                     image_gray = cv.cvtColor(image_data, cv.COLOR_RGB2GRAY)
#                     image_data = cv.cvtColor(image_gray, cv.COLOR_GRAY2RGB)
#
#         image_data = image_data / 255
#         mask_data = np.array(new_mask)
#
#         index = index + 1
#         bbox_data = []
#         # 对bbox进行重新处理
#         if len(bbox) > 0:
#             # 根据新的宽高与原来宽高的比例得到缩放后的坐标，再加上这张图片在新的大图上的左上角坐标
#             bbox[:, [0, 2]] = bbox[:, [0, 2]] * nw / iw + dx
#             bbox[:, [1, 3]] = bbox[:, [1, 3]] * nh / ih + dy
#
#             raw_bbox = copy.deepcopy(bbox)
#
#             # 小于0的都变成0，超过w,h的bbox值都变成w,h
#             bbox[:, 0:2][bbox[:, 0:2] < 0] = 0
#
#             ##超过像素索引,255之后会从0开始,-1已经到像素的最大索引了
#             bbox[:, 2][bbox[:, 2] > (w - 1)] = w - 1
#             bbox[:, 3][bbox[:, 3] > (h - 1)] = h - 1
#
#             # 计算前后的面积比例,保留大于一定比例的标注
#             now_areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
#             raw_areas = (raw_bbox[:, 2] - raw_bbox[:, 0]) * (raw_bbox[:, 3] - raw_bbox[:, 1])
#
#             save_index = np.where(now_areas / raw_areas > 0.95)[0]
#             bbox = bbox[save_index]
#             # ** "用面积过滤指定类别对象"
#             # both_save_inds=np.array([])
#             # bask_inds=np.where(bbox[:,4]==bask_num)[0]
#             # bask_save_inds=np.where(now_areas[bask_inds]>3200)[0]
#             # both_save_inds=both_save_inds+bask_inds[bask_save_inds]
#             #
#             # trace_inds = np.where(bbox[:, 4] == trace_num)[0]
#             # trace_save_inds = np.where(now_areas[trace_inds] > 43000)[0]
#             # both_save_inds = both_save_inds + trace_inds[trace_save_inds]
#             #
#             # bbox=bbox[both_save_inds]
#
#             if len(bbox) > 0:
#                 bbox_w = bbox[:, 2] - bbox[:, 0]
#                 bbox_h = bbox[:, 3] - bbox[:, 1]
#                 # 需要bbox的宽高都大于1的bbox才能进入下一步，and表示为与操作，两者都正确则结果正确,bbox[True,True,True...]
#                 bbox = bbox[np.logical_and(bbox_w > 1, bbox_h > 1)]
#
#                 bbox_data = np.zeros((len(bbox), 5))
#                 bbox_data[:len(bbox)] = bbox
#
#         image_datas.append(image_data)
#         mask_datas.append(mask_data)
#         bbox_datas.append(bbox_data)
#
#         # **mask_show = Image.fromarray((mask_data * 255).astype(np.uint8))
#         # for j in range(len(bbox_data)):
#         #
#         #     left, top, right, bottom = bbox_data[j][0:4]
#         #     draw = ImageDraw.Draw(mask_show)
#         #     draw.rectangle([left , top , right, bottom ], outline="green",width=2)
#         #     # #thickness这里控制了i的值，会使得bbox相对于真实值往内收缩
#         #     # thickness = 3
#         #     # for i in range(thickness):
#         #     #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
#         # mask_show.show()
#         # **print("fsdff")
#
#         # **img = Image.fromarray((image_data * 255).astype(np.uint8))
#
#         # **img.show()
#         # 经过以上处理就是把一张图片处理完了
#
#     # 将图片分割，放在一起
#     cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
#     cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
#
#     new_image = np.zeros([h, w, 3])
#     new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
#     new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
#     new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
#     new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
#
#     new_image = (new_image * 255).astype(np.uint8)
#
#     # **#检验第四个顺序的图bbox对应标签
#     # mmask=Image.fromarray((mask_datas[3]*255).astype(np.uint8))
#     # for num in range(len(bbox_datas[3])):
#     #     left, top, right, bottom = bbox_datas[3][num]
#     #     draw = ImageDraw.Draw(mmask)
#     #     draw.rectangle([left , top , right, bottom ], outline="green",width=2)
#     #
#     # **mmask.show()
#
#     new_mask = np.zeros([h, w, 3])
#     new_mask[:cuty, :cutx, :] = mask_datas[0][:cuty, :cutx, :]
#     new_mask[cuty:, :cutx, :] = mask_datas[1][cuty:, :cutx, :]
#     new_mask[cuty:, cutx:, :] = mask_datas[2][cuty:, cutx:, :]
#     new_mask[:cuty, cutx:, :] = mask_datas[3][:cuty, cutx:, :]
#     new_mask = new_mask.astype(np.uint8)
#
#     new_bbox = merge_bboxes(bbox_datas, cutx, cuty, h, w, save_proportion=save_proportion)
#     # if len(new_bbox)>0:
#     #     **"合并bbox后,再次用面积过滤指定类别对象"
#     #     new_bbox=np.array(new_bbox)
#     #     now_areas = (new_bbox[:, 2] - new_bbox[:, 0]) * (new_bbox[:, 3] - new_bbox[:, 1])
#     #
#     #     both_save_inds = np.array([])
#     #     bask_inds = np.where(new_bbox[:, 4] == bask_num)[0]
#     #     bask_save_inds = np.where(now_areas[bask_inds] > 2000)[0]
#     #     # bask_save_inds = np.where(now_areas[bask_inds] > 3200)[0]
#     #     both_save_inds = np.concatenate((both_save_inds , bask_inds[bask_save_inds]),axis=0)
#     #
#     #     trace_inds = np.where(new_bbox[:, 4] == trace_num)[0]
#     #     trace_save_inds = np.where(now_areas[trace_inds] > 16000)[0]
#     #     both_save_inds = np.concatenate((both_save_inds , trace_inds[trace_save_inds]),axis=0)
#     #     both_save_inds=both_save_inds.astype(np.uint8)
#     #     if len(both_save_inds)>0:
#     #          new_bbox = list(new_bbox[both_save_inds])
#     #     else:
#     #         return  [],[],[]
#     if len(new_bbox) > 0:
#         new_bbox = np.array(new_bbox).astype(np.uint16)
#         # [xmin:xmax,ymin:ymax]没取到xmax,ymax
#         new_bbox[:, 2] = new_bbox[:, 2] + 1
#         new_bbox[:, 3] = new_bbox[:, 3] + 1
#
#         new_bbox[:, 2][new_bbox[:, 2] > w - 1] = w - 1
#         new_bbox[:, 3][new_bbox[:, 3] > h - 1] = h - 1
#
#         # **for j in range(len(new_bbox)):
#         #     left, top, right, bottom = new_bbox[j][0:4]
#         #     draw = ImageDraw.Draw(show_mask)
#         #     draw.rectangle([left, top, right, bottom], outline="red", width=1)
#         #
#         # **show_mask.show()
#
#         return new_image, new_mask, new_bbox
#     else:
#         return [], [], []

