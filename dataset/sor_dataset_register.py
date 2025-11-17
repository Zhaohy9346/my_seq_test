import os, json
from detectron2.data import DatasetCatalog, MetadataCatalog

def loadJson(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def prepare_sor_dataset_list_of_dict(dataset_id, split, root="datasets"):
    path_to_ds = os.path.join(root, dataset_id, "{}_{}.json".format(dataset_id, split))
    print("Path to {}: {}".format(dataset_id, path_to_ds), flush=True)

    dataset = loadJson(path_to_ds)
    print(dataset.get("__comment__", "No comment"), flush=True)

    image_path = os.path.join(root, dataset_id, "images", split)
    dataset_data = dataset["data"]

    for i in range(len(dataset_data)):
        dataset_data[i]["file_name"] = os.path.join(image_path, dataset_data[i]["image_name"]+".jpg")
    print("#Length of SOR dataset [{}]:{}".format(dataset_id, len(dataset_data)), flush=True)
    
    return dataset_data

# ===== 添加新的COD数据集加载函数 =====
def prepare_cod_dataset_list_of_dict(split, root="/data1/zhy/CODdata/rank"):
    """
    加载COD数据集
    Args:
        split: 'train', 'val', or 'test'
        root: 数据集根目录
    """
    # JSON文件路径
    if split == "train":
        json_file = os.path.join(root, split, "cor_dataset_detectron2-tr.json")
    elif split == "val":
        json_file = os.path.join(root, split, "cor_dataset_detectron2-val.json")
    else:  # test
        json_file = os.path.join(root, split, "cor_dataset_detectron2-te.json")
    
    print("Path to COD dataset: {}".format(json_file), flush=True)
    
    # 加载JSON数据
    dataset = loadJson(json_file)
    
    # 图片目录
    image_path = os.path.join(root, split, "img")
    
    # 处理数据格式
    dataset_data = []
    for item in dataset:
        # 创建符合detectron2格式的数据项
        data_item = {
            "image_name": item["image_name"],
            "file_name": os.path.join(image_path, item["image_name"] + ".jpg"),  # 确认扩展名
            "height": item["height"],
            "width": item["width"],
            "image_id": item.get("image_id", len(dataset_data)),
            "annotations": []
        }
        
        # 处理annotations
        # 根据rank值排序，确保rank值转换为category_id
        annotations = item.get("annotations", [])
        
        # 如果rank是0.0，需要根据实际排序来分配category_id
        # 假设标注已经按显著性排序，第一个最显著
        for idx, ann in enumerate(annotations):
            new_ann = {
                "segmentation": ann["segmentation"],
                "bbox": ann.get("bbox", []),
                "bbox_mode": ann.get("bbox_mode", 1),
                # category_id表示排名：1=最显著，2=次显著，...
                "category_id": idx + 1,  # 或者使用 ann.get("rank", idx+1)
                "rank": ann.get("rank", idx + 1)
            }
            data_item["annotations"].append(new_ann)
        
        dataset_data.append(data_item)
    
    print("#Length of COD dataset [{}]: {}".format(split, len(dataset_data)), flush=True)
    
    return dataset_data

# ===== 更健壮的版本，处理不同的rank值 =====
def prepare_cod_dataset_with_rank_processing(split, root="/data1/zhy/CODdata/rank"):
    """
    加载COD数据集，并正确处理rank值
    """
    if split == "train":
        json_file = os.path.join(root, split, "cor_dataset_detectron2-tr.json")
    elif split == "val":
        json_file = os.path.join(root, split, "cor_dataset_detectron2-val.json")
    else:
        json_file = os.path.join(root, split, "cor_dataset_detectron2-te.json")
    
    print("Path to COD dataset: {}".format(json_file), flush=True)
    
    dataset = loadJson(json_file)
    image_path = os.path.join(root, split, "img")
    
    dataset_data = []
    skipped_images = 0
    
    for item in dataset:
        data_item = {
            "image_name": item["image_name"],
            "file_name": os.path.join(image_path, item["image_name"] + ".jpg"),
            "height": item["height"],
            "width": item["width"],
            "image_id": item.get("image_id", len(dataset_data)),
            "annotations": []
        }
        
        annotations = item.get("annotations", [])
        
        # ====== 关键修改：重新分配category_id ======
        # 不管原始的category_id或rank是什么，按顺序重新分配
        # 假设annotations已经按显著性排序，第一个最显著
        
        # 如果标注太多，限制数量以避免内存问题
        MAX_ANNOTATIONS = 10  # 限制最多10个对象
        if len(annotations) > MAX_ANNOTATIONS:
            print(f"Warning: Image {item['image_name']} has {len(annotations)} annotations, "
                  f"truncating to {MAX_ANNOTATIONS}")
            annotations = annotations[:MAX_ANNOTATIONS]
        
        for idx, ann in enumerate(annotations):
            new_ann = {
                "segmentation": ann["segmentation"],
                "bbox": ann.get("bbox", []),
                "bbox_mode": ann.get("bbox_mode", 1),
                # 重新分配category_id为 1, 2, 3, ...
                "category_id": idx + 1,  # 1-based ranking
                "rank": idx + 1,
                "original_category_id": ann.get("category_id", 0),  # 保存原始值
                "original_rank": ann.get("rank", 0)
            }
            data_item["annotations"].append(new_ann)
        
        # 如果没有有效标注，跳过这张图
        if len(data_item["annotations"]) == 0:
            skipped_images += 1
            print(f"Warning: Skipping image {item['image_name']} - no valid annotations")
            continue
        
        dataset_data.append(data_item)
    
    print(f"#Length of COD dataset [{split}]: {len(dataset_data)}")
    print(f"#Skipped images: {skipped_images}")
    
    return dataset_data

def register_sor_dataset(cfg):
    # 原有数据集
    DatasetCatalog.register("assr_train", lambda s="train":prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_val", lambda s="val":prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_test", lambda s="test":prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s, root=cfg.DATASETS.ROOT))

    DatasetCatalog.register("irsr_train", lambda s="train":prepare_sor_dataset_list_of_dict(dataset_id="irsr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("irsr_test", lambda s="test":prepare_sor_dataset_list_of_dict(dataset_id="irsr", split=s, root=cfg.DATASETS.ROOT))
    
    # ===== 注册COD数据集 =====
    DatasetCatalog.register("cod_train", lambda: prepare_cod_dataset_with_rank_processing(split="train"))
    DatasetCatalog.register("cod_val", lambda: prepare_cod_dataset_with_rank_processing(split="val"))
    DatasetCatalog.register("cod_test", lambda: prepare_cod_dataset_with_rank_processing(split="test"))