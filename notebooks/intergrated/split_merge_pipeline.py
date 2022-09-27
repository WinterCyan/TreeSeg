"""
dataset preprocessing, split, (prediction), merge pipeline
1. input: pan.tif, ndvi.tif
2. tif -> read, normalization(BEFORE/AFTER split), write as png
3. split, assign index, save as datdaset
4. load dataset and inference
5. merge results, annotation corresponding to img
6. convert to shp, count tree
"""

from operator import contains
import numpy as np
import rasterio
import os

def image_normalize(im, axis = (0,1), c = 1e-8):
    return (im - im.mean(axis)) / (im.std(axis) + c)

def read_tif(dir):
    pass

    """

    Args:
        tif_dir: folder contains ALL pan-ndvi tif pairs
        sample_dir: folder contains ALL training/inference samples

    """

def split_samples(tif_dir, sample_dir, split_unit, mode, area_polygon_dir, norm_mode):
    """read tif & split into pngs

    Args:
        tif_dir: folder contains ALL pan-ndvi tif pairs
        sample_dir: folder contains ALL training/inference samples
        split_unit: pixel size of split square
        mode: ["train", "inference"], for split training data or inference data
        area_polygon_dir (_type_): if mode="train", area_polygon_dir NOT null
    """

    tif_dir = tif_dir.rstrip("/")
    sample_dir = sample_dir.rstrip("/")
    all_pan_files_names = [name for name in os.listdir(tif_dir) if (name.__contains__("pan") or name.__contains__("PAN")) and name.endswith(".tif")]
    all_ndvi_files_names = [name for name in os.listdir(tif_dir) if (name.__contains__("ndvi") or name.__contains__("NDVI")) and name.endswith(".tif")]
    for pan_item in all_pan_files_names:
        assert all_ndvi_files_names.count(pan_item.replace("pan","ndvi")) or all_ndvi_files_names.count(pan_item.replace("PAN","NDVI")), "cannot pair up all tif pan & ndvi images!"
    print(f"found {len(all_pan_files_names)} pairs of tif images in {tif_dir} ...")
    for pan_item in all_pan_files_names:
        pan_full_path = f"{tif_dir}/{pan_item}"
        ndvi_item = ""
        if pan_item.__contains__("pan"):
            ndvi_item = pan_item.replace("pan", "ndvi")
        elif pan_item.__contains__("PAN"):
            ndvi_item = pan_item.replace("PAN", "NDVI")
        assert ndvi_item != ""
        ndvi_full_path = f"{tif_dir}/{ndvi_item}"
        print(pan_full_path, ndvi_full_path)

        pan_dataset = rasterio.open(pan_full_path)
        pan_profile = pan_dataset.profile
        ndvi_dataset = rasterio.open(ndvi_full_path)
        ndvi_profile = ndvi_dataset.profile

        pan_profile['dtype'] = rasterio.float32
        pan_profile['height'] = split_unit
        pan_profile['width'] = split_unit
        # pan_profile['transform'] = pan_dataset
        pan_profile['blockxsize'] = 32
        pan_profile['blockysize'] = 32
        pan_profile['count'] = 1

        ndvi_profile['dtype'] = rasterio.float32
        ndvi_profile['height'] = split_unit
        ndvi_profile['width'] = split_unit
        # ndvi_profile['transform'] = ndvi_dataset
        ndvi_profile['blockxsize'] = 32
        ndvi_profile['blockysize'] = 32
        ndvi_profile['count'] = 1

        pan_arr = pan_dataset.read(1).astype(pan_profile['dtype'])
        ndvi_arr = ndvi_dataset.read(1).astype(ndvi_profile['dtype'])
        if norm_mode=="before_split":
            pan_arr = image_normalize(pan_arr)
            ndvi_arr = image_normalize(ndvi_arr)
        height, width = pan_arr.shape
        print(f"read tif, shape: {height}x{width}")

        split_rows = int(height/split_unit)
        split_cols = int(width/split_unit)
        print(f"spliting into {split_rows}x{split_cols} squares...")
        for r in range(split_rows):
            for c in range(split_cols):
                idx_str = f"r{r}-c{c}"
                print(f"writing {idx_str}...")
                pan_sample = pan_arr[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                ndvi_sample = ndvi_arr[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                if norm_mode=="after_split":
                    pan_sample = image_normalize(pan_sample)
                    ndvi_sample = image_normalize(ndvi_sample)
                with rasterio.open(f"{sample_dir}/{idx_str}-{pan_item.replace('tif','png')}", 'w', **pan_profile) as dst:
                    dst.write(pan_sample, 1)
                    print(pan_sample.shape)
                    print(f"max: {np.max(pan_sample)}, min: {np.min(pan_sample)}")
                    dst.close()
                with rasterio.open(f"{sample_dir}/{idx_str}-{ndvi_item.replace('tif','png')}", 'w', **ndvi_profile) as dst:
                    dst.write(ndvi_sample, 1)
                    dst.close()



if __name__ == '__main__':
    split_samples(
            tif_dir="/Users/wintercyan/LocalDocuments/treeseg-resource/test/data/", 
            sample_dir="/Users/wintercyan/LocalDocuments/treeseg-resource/test/png_dataset/",
            split_unit=160,
            mode="",
            area_polygon_dir="",
            norm_mode="after_split"
        )
    # inference()
    # merge_results()