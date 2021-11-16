import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]

        row, col = paste_img.shape[:2]
        img[:row, :col] = paste_img * alpha + img[:row, :col] * (1 - alpha)

        img = img.astype(img_dtype)

    return img


def mask_copy_paste(mask, paste_mask, alpha):
    raise NotImplementedError


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


def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        # change the alpha to accept any size paste mask
        # show_two_image(alpha,paste_masks)
        # masks = [
        #     np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks  #just like image ,left-up overlap,masks are in the overlap
        # ]
        # masks.extend(paste_masks)

        row, col = alpha.shape[:2]
        for mask in masks:
            portion_mask = mask[:row, :col]
            portion_mask = np.logical_and(portion_mask, np.logical_xor(portion_mask, alpha)).astype(np.uint8)
            mask[:row, :col] = portion_mask

        new_paste_masks = []
        for p_mask in paste_masks:
            zero_mask = np.zeros_like(masks[0], dtype=np.uint8)
            zero_mask[:row, :col] = p_mask
            new_paste_masks.append(zero_mask)

        masks.extend(new_paste_masks)

    return masks


def extract_bboxes(masks):
    bboxes = []
    # allow for cases of no mask
    if len(masks) == 0:
        return bboxes

    h, w = masks[0].shape
    for mask in masks:
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
            y1 /= w
            y2 /= w
            x1 /= h
            x2 /= h
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0

        bboxes.append((y1, x1, y2, x2))

    return bboxes


def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha, key):
    if key == 'paste_bboxes':
        return bboxes
    elif paste_bboxes is not None:
        # masks = masks_copy_paste(masks, paste_masks=[], alpha=alpha)  #个人觉得多余,还会抹去paste上去的mask
        adjusted_bboxes = extract_bboxes(masks)

        # only keep the bounding boxes for objects listed in bboxes
        mask_indices = [box[-1] for box in bboxes]
        adjusted_bboxes = [adjusted_bboxes[idx] for idx in mask_indices]
        # append bbox tails (classes, etc.)
        adjusted_bboxes = [bbox + tail[4:] for bbox, tail in zip(adjusted_bboxes, bboxes)]

        # adjust paste_bboxes mask indices to avoid overlap
        if len(masks) > 0:
            max_mask_index = len(masks)
        else:
            max_mask_index = 0
        # 从mask里提取bbox后, 把对应的类别id也加上去
        paste_mask_indices = [max_mask_index + ix for ix in range(len(paste_bboxes))]
        paste_bboxes = [pbox[:-1] + (pmi,) for pbox, pmi in zip(paste_bboxes, paste_mask_indices)]
        adjusted_paste_bboxes = extract_bboxes(paste_masks)
        adjusted_paste_bboxes = [apbox + tail[4:] for apbox, tail in zip(adjusted_paste_bboxes, paste_bboxes)]

        bboxes = adjusted_bboxes + adjusted_paste_bboxes

    return bboxes


def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    # remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)

        keypoints = visible_keypoints + paste_keypoints

    return keypoints


class CopyPaste(A.DualTransform):
    def __init__(
            self,
            blend=True,
            sigma=3,
            pct_objects_paste=0.1,
            max_paste_objects=None,
            p=0.5,
            always_apply=False
    ):
        super(CopyPaste, self).__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects
        self.p = p
        self.always_apply = always_apply

    @staticmethod
    def get_class_fullname():
        return 'copypaste.CopyPaste'

    @property
    def targets_as_params(self):
        return [
            "masks",
            "paste_image",
            # "paste_mask",
            "paste_masks",
            "paste_bboxes",
            # "paste_keypoints"
        ]

    def get_params_dependent_on_targets(self, params):
        image = params["paste_image"]
        masks = None
        if "paste_mask" in params:
            # handle a single segmentation mask with
            # multiple targets
            # nothing for now.
            raise NotImplementedError
        elif "paste_masks" in params:
            masks = params["paste_masks"]

        assert (masks is not None), "Masks cannot be None!"

        bboxes = params.get("paste_bboxes", None)
        keypoints = params.get("paste_keypoints", None)

        # number of objects: n_bboxes <= n_masks because of automatic removal
        n_objects = len(bboxes) if bboxes is not None else len(masks)

        # paste all objects if no restrictions
        n_select = n_objects
        if self.pct_objects_paste:
            n_select = int(n_select * self.pct_objects_paste)
        if self.max_paste_objects:
            n_select = min(n_select, self.max_paste_objects)

        # no objects condition
        if n_select == 0:
            return {
                "param_masks": params["masks"],
                "paste_img": None,
                "alpha": None,
                "paste_mask": None,
                "paste_masks": None,
                "paste_bboxes": None,
                "paste_keypoints": None,
                "objs_to_paste": []
            }

        # select objects,从总的bbox里随机选取n_select个，返回的是它们的索引号
        objs_to_paste = np.random.choice(
            range(0, n_objects), size=n_select, replace=False
        )

        # take the bboxes
        if bboxes:
            bboxes = [bboxes[idx] for idx in objs_to_paste]
            # the last label in bboxes is the index of corresponding mask
            mask_indices = [bbox[-1] for bbox in bboxes]

        # create alpha by combining all the objects into
        # a single binary mask
        masks = [masks[idx] for idx in mask_indices]

        alpha = masks[0] > 0
        for mask in masks[1:]:
            alpha += mask > 0

        return {
            "param_masks": params["masks"],
            "paste_img": image,
            "alpha": alpha,
            "paste_mask": None,
            "paste_masks": masks,
            "paste_bboxes": bboxes,
            "paste_keypoints": keypoints
        }

    @property
    def ignore_kwargs(self):
        return [
            "paste_image",
            "paste_mask",
            "paste_masks"
        ]

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None and key not in self.ignore_kwargs:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                target_dependencies['key'] = key
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def apply(self, img, paste_img, alpha, **params):
        return image_copy_paste(
            img, paste_img, alpha, blend=self.blend, sigma=self.sigma
        )

    def apply_to_mask(self, mask, paste_mask, alpha, **params):
        return mask_copy_paste(mask, paste_mask, alpha)

    def apply_to_masks(self, masks, paste_masks, alpha, **params):
        return masks_copy_paste(masks, paste_masks, alpha)

    def apply_to_bboxes(self, bboxes, paste_bboxes, param_masks, paste_masks, alpha, key, **params):
        return bboxes_copy_paste(bboxes, paste_bboxes, param_masks, paste_masks, alpha, key)

    def apply_to_keypoints(self, keypoints, paste_keypoints, alpha, **params):
        raise NotImplementedError
        # return keypoints_copy_paste(keypoints, paste_keypoints, alpha)

    def get_transform_init_args_names(self):
        return (
            "blend",
            "sigma",
            "pct_objects_paste",
            "max_paste_objects"
        )


def get_mask_bbox(mask):
    """
    获取mask中所有mask的最小外接矩形，所以如果图中多个对象分布的范围很广，那就不太好弄
    """
    from skimage import measure

    contours = measure.find_contours(mask, 0.5)

    bbox_list = []

    for idx in range(len(contours)):
        contour = contours[idx]
        contour = np.flip(contour, axis=1)

        arr_seg = np.expand_dims(contour, axis=1)
        arr_seg = np.array(arr_seg, dtype=np.int)
        x, y, w, h = cv2.boundingRect(arr_seg)
        bbox_list.append([int(x), int(y), int(x + w), int(y + h), 11])

    if len(bbox_list) > 1:
        raise ("Length of bbox errors")

    return bbox_list[0]


def checkout_paste_bbox(row, column, paste_img_data):
    if paste_img_data['image'].shape[0] > row or paste_img_data['image'].shape[1] > column:
        paste_img_data = resize_image_mask_bbox(row, column, paste_img_data)

    # 把非稀有类别从paste里过滤
    bboxes = np.array(paste_img_data['bboxes'], dtype=np.int)
    cate_ids, indexs = bboxes[:, -2], bboxes[:, -1]

    target_catid = [13, 14, 17, 16, 18, 19]

    save_indexs = []
    for cat, ind in zip(cate_ids, indexs):
        if cat in target_catid:
            save_indexs.append(ind)

    new_masks = []
    new_bboxes = []
    for ind, box_ind in zip(save_indexs, range(len(save_indexs))):
        new_masks.append(paste_img_data['masks'][ind])
        new_bboxes.append(paste_img_data['bboxes'][ind][:5] + [box_ind])

    paste_img_data['bboxes'] = new_bboxes
    paste_img_data['masks'] = new_masks

    return paste_img_data


def resize_image_mask_bbox(row, column, paste_img_data):
    p_row, p_col = paste_img_data['image'].shape[:2]
    ratio = min(column / p_col, row / p_row) - 0.0001
    paste_img_data['image'] = cv2.resize(paste_img_data['image'],
                                         (int(ratio * p_col), int(ratio * p_row)))  # resize image

    for mask_id in range(len(paste_img_data['masks'])):  # resize masks
        paste_img_data['masks'][mask_id] = cv2.resize(paste_img_data['masks'][mask_id],
                                                      (int(ratio * p_col), int(ratio * p_row)))

    for box_id in range(len(paste_img_data['bboxes'])):  # resize bboxes
        paste_img_data['bboxes'][box_id] = [int(x * ratio) for x in paste_img_data['bboxes'][box_id][:4]] + \
                                           [int(x) for x in paste_img_data['bboxes'][box_id][4:]]

    return paste_img_data


def copy_paste_class(dataset_class):
    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index + 1:]

            # replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            # recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def __getitem__raw(self, idx):
        # 代码原始的getitem，但是为了去增强长尾数据集，所以先不用这个
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()

        img_data = self.load_example(idx)
        if self.copy_paste is not None:
            while True:
                # 随机选取一张图片，将上面的对象粘贴到另一张图上
                paste_idx = random.randint(0, self.__len__() - 1)
                paste_img_data = self.load_example(paste_idx)

                # 这里没有缩放，是直接把两个图叠加到一起的，所以会有一些paste图片的bbox超过copy图片
                paste_img_data = checkout_paste_bbox(img_data['image'].shape[0], img_data['image'].shape[1],
                                                     paste_img_data)
                if paste_img_data is not None:
                    break

            for k in list(paste_img_data.keys()):
                paste_img_data['paste_' + k] = paste_img_data[k]
                del paste_img_data[k]

            img_data = self.copy_paste(**img_data, **paste_img_data)
            img_data = self.post_transforms(**img_data)
            img_data['paste_index'] = paste_idx

        return img_data

    def __getitem__(self, idx):
        # 应用于长尾数据集的copy-paste
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()

        img_data = self.load_example(idx)
        if self.copy_paste is not None:

            # 从尾类图片中选取一张，粘贴到copy图片上
            paste_idx, paste_num = get_paste_index()
            paste_img_data = self.load_example(paste_idx)

            # 这里没有缩放，是直接把两个图叠加到一起的，所以会有一些paste图片的bbox超过copy图片
            paste_img_data = checkout_paste_bbox(img_data['image'].shape[0], img_data['image'].shape[1],
                                                 paste_img_data)

            for k in list(paste_img_data.keys()):
                paste_img_data['paste_' + k] = paste_img_data[k]
                del paste_img_data[k]

            img_data = self.copy_paste(**img_data, **paste_img_data)
            img_data = self.post_transforms(**img_data)
            img_data['paste_index'] = paste_idx

        return img_data

    setattr(dataset_class, '_split_transforms', _split_transforms)
    setattr(dataset_class, '__getitem__', __getitem__)

    return dataset_class


def get_paste_index():
    ""
    txt_path = r"/home/data1/yw/copy_paste_empty/500_aug/hrsc_104_tv_raw_trans/train_data/aug_fold_v1/process/test_18.txt"
    with open(txt_path, 'r') as f:
        copy_indexs = f.readlines()
    copy_indexs = [int(x.strip('\n')) for x in copy_indexs]
    random_id = random.randint(0, len(copy_indexs) - 1)
    index = copy_indexs[random_id]

    return index, len(copy_indexs)
