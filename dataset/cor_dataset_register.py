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
    print(dataset["__comment__"], flush=True)

    image_path = os.path.join(root, dataset_id, "images", split)
    dataset_data = dataset["data"]

    for i in range(len(dataset_data)):
        dataset_data[i]["file_name"] = os.path.join(image_path, dataset_data[i]["image_name"]+".jpg")
    print("#Length of SOR dataset [{}]:{}".format(dataset_id, len(dataset_data)), flush=True)
    
    return dataset_data

# ===== 添加新的数据集加载函数 =====
def prepare_cod_dataset_list_of_dict(split, root="/data/zhaohy/CODdata/rank"):
    """
    加载COD数据集
    Args:
        split: 'train', 'val', or 'test'
        root: 数据集根目录
    """
    # 根据你的文件命名规则，可能需要调整json文件名
    json_file = os.path.join(root, split, "cor_dataset_detectron2-{}.json".format("tr" if split=="train" else split))
    print("Path to COD dataset: {}".format(json_file), flush=True)
    
    dataset = loadJson(json_file)
    if "__comment__" in dataset:
        print(dataset["__comment__"], flush=True)
    
    image_path = os.path.join(root, split, "img")
    dataset_data = dataset["data"] if "data" in dataset else dataset["images"]
    
    for i in range(len(dataset_data)):
        # 根据你的json格式调整，可能需要修改字段名
        img_name = dataset_data[i].get("image_name") or dataset_data[i].get("file_name")
        # 确保图片扩展名正确，可能是.jpg, .png等
        dataset_data[i]["file_name"] = os.path.join(image_path, img_name)
    
    print("#Length of COD dataset [{}]:{}".format(split, len(dataset_data)), flush=True)
    
    return dataset_data

def register_sor_dataset(cfg):
    # 原有的数据集注册
    DatasetCatalog.register("assr_train", lambda s="train":prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_val", lambda s="val":prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_test", lambda s="test":prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s, root=cfg.DATASETS.ROOT))

    DatasetCatalog.register("irsr_train", lambda s="train":prepare_sor_dataset_list_of_dict(dataset_id="irsr", split=s, root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("irsr_test", lambda s="test":prepare_sor_dataset_list_of_dict(dataset_id="irsr", split=s, root=cfg.DATASETS.ROOT))
    
    # ===== 注册新的COD数据集 =====
    DatasetCatalog.register("cod_train", lambda: prepare_cod_dataset_list_of_dict(split="train"))
    DatasetCatalog.register("cod_val", lambda: prepare_cod_dataset_list_of_dict(split="val"))
    DatasetCatalog.register("cod_test", lambda: prepare_cod_dataset_list_of_dict(split="test"))