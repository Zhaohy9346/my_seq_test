import copy
import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A


def read_image(file_name, format="RGB"):
    return np.array(Image.open(file_name).convert(format)).astype(np.uint8)


def parse_anno(anno, H, W, max_polygons=50):
    """
    解析annotation并创建mask
    限制polygon数量为50个以避免极端情况
    """
    mask = np.zeros((H, W), dtype=float)
    
    try:
        if "segmentation" not in anno or not anno["segmentation"]:
            return mask
        
        segmentations = anno["segmentation"]
        
        if not isinstance(segmentations, list):
            return mask
        
        # 如果polygon数量超过上限，只保留面积最大的前N个
        if len(segmentations) > max_polygons:
            # 计算每个polygon的面积
            polygon_with_area = []
            for seg in segmentations:
                if isinstance(seg, list) and len(seg) >= 6:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    try:
                        area = cv2.contourArea(poly)
                        polygon_with_area.append((area, seg))
                    except:
                        continue
            
            # 按面积降序排序，取前max_polygons个
            polygon_with_area.sort(key=lambda x: x[0], reverse=True)
            segmentations = [seg for area, seg in polygon_with_area[:max_polygons]]
            
            print(f"Warning: Annotation had {len(anno['segmentation'])} polygons, "
                  f"reduced to {len(segmentations)} largest ones")
        
        # 解析有效的polygons
        valid_polygons = []
        for seg in segmentations:
            if isinstance(seg, list) and len(seg) >= 6:
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                if len(poly) >= 3:
                    valid_polygons.append(poly)
        
        if valid_polygons:
            cv2.fillPoly(mask, valid_polygons, 1.0)
        
    except Exception as e:
        print(f"Warning: Error parsing annotation: {e}")
    
    return mask


def merge_masks(masks, H, W):
    mask = np.zeros((H, W), dtype=float)
    for m in masks:
        mask += m
    return np.where(mask > 0.5, 1.0, 0.0)


def sampleRankExp(ranks):
    num_level = max(ranks)
    target = int(np.log2(np.random.randint(0, int(2 ** num_level)) + 1)) + 1
    target = 0 if target > num_level else target
    return target


def sor_dataset_mapper_train(dataset_dict, cfg):
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # 读取图像
    try:
        image = read_image(dataset_dict["file_name"], format="RGB")
    except Exception as e:
        print(f"Error reading image {dataset_dict.get('file_name', 'unknown')}: {e}")
        raise
    
    H, W = dataset_dict["height"], dataset_dict["width"]
    
    # 解析annotations
    ranks = []
    masks = []
    
    for anno in dataset_dict["annotations"]:
        cate = anno.get("category_id", 0)
        if cate > 0:
            # parse_anno 现在会自动处理多个polygon
            mask = parse_anno(anno, H, W)
            
            # 检查mask总面积
            mask_area = mask.sum()
            if mask_area > 10:  # 至少10个像素
                ranks.append(cate)
                masks.append(mask)
            else:
                print(f"Warning: Skipping mask with total area {mask_area} for category {cate} "
                      f"in image {dataset_dict.get('image_name', 'unknown')}")
                print(f"  Number of polygons: {len(anno.get('segmentation', []))}")
    
    # 如果没有有效mask，创建一个默认mask
    if len(masks) == 0:
        print(f"WARNING: No valid masks for image {dataset_dict.get('image_name', 'unknown')}. "
              f"Creating default mask.")
        dummy_mask = np.zeros((H, W), dtype=float)
        h_start, h_end = H // 4, 3 * H // 4
        w_start, w_end = W // 4, 3 * W // 4
        dummy_mask[h_start:h_end, w_start:w_end] = 1.0
        masks.append(dummy_mask)
        ranks.append(1)
    
    # 按category_id排序
    gts = list((r, m) for r, m in zip(ranks, masks))
    gts.sort(key=lambda x: x[0], reverse=False)  # 小的rank在前
    ranks = [x[0] for x in gts]
    masks = [x[1] for x in gts]

    # 数据增强
    try:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(cfg.INPUT.FT_SIZE_TRAIN, cfg.INPUT.FT_SIZE_TRAIN)
        ],
            additional_targets=dict(("mask_{}".format(i), "mask") for i in range(len(masks)))
        )
        aug = transform(image=image, **dict(("mask_{}".format(i), masks[i]) for i in range(len(masks))))
        image = aug["image"]
        masks = [aug["mask_{}".format(i)] for i in range(len(masks))]
    except Exception as e:
        print(f"Error in data augmentation: {e}")
        raise

    # 转换为Tensor
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    masks = torch.stack([torch.from_numpy(m).float() for m in masks], dim=0)

    return {
        "image_name": dataset_dict.get("image_name", "unknown"),
        "image": image,
        "height": H,
        "width": W,
        "masks": masks,
        "ranks": ranks
    }