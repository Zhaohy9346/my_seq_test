import sys
sys.path.append(".")

import json
import numpy as np

def analyze_dataset(json_file):
    """分析数据集中的polygon分布"""
    
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Analyzing: {json_file}")
    print(f"Total images: {len(dataset)}")
    
    total_annotations = 0
    total_polygons = 0
    multi_polygon_annotations = 0
    polygon_counts = []
    
    max_polygons = 0
    max_polygons_image = None
    
    for sample in dataset:
        for ann in sample['annotations']:
            total_annotations += 1
            
            num_polygons = len(ann.get('segmentation', []))
            total_polygons += num_polygons
            polygon_counts.append(num_polygons)
            
            if num_polygons > 1:
                multi_polygon_annotations += 1
            
            if num_polygons > max_polygons:
                max_polygons = num_polygons
                max_polygons_image = sample['image_name']
    
    print(f"\nAnnotation Statistics:")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Total polygons: {total_polygons}")
    print(f"  Annotations with multiple polygons: {multi_polygon_annotations} "
          f"({100*multi_polygon_annotations/total_annotations:.1f}%)")
    print(f"  Average polygons per annotation: {np.mean(polygon_counts):.2f}")
    print(f"  Max polygons in one annotation: {max_polygons} (image: {max_polygons_image})")
    
    print(f"\nPolygon count distribution:")
    unique, counts = np.unique(polygon_counts, return_counts=True)
    for num, count in zip(unique, counts):
        print(f"  {num} polygon(s): {count} annotations ({100*count/total_annotations:.1f}%)")

if __name__ == "__main__":
    analyze_dataset("/data1/zhy/CODdata/rank/train/cor_dataset_detectron2-tr.json")