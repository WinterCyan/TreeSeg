# ALERT: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

data_root="/home/winter/code-resource/treeseg/trainingdata"
proj_root="/home/winter/code/TreeSeg"

# -------------------------- preprocess --------------------------
python3 -W ignore ../split_infer_merge_pipeline.py \
    --task preprocess_train \
    --tif_dir $data_root/tif \
    --area_polygon_dir $data_root/shp \
    --area_range list \
    --area_idx 0 1 2 4 5 8 9 12 13 14 15 16 17 18 19 20 21 22 23 24 25 28 29 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 \
    --interm_png_dir $data_root/interm_png
