# ALERT: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

data_root="/home/winter/code-resource/treeseg/trainingdata"
proj_root="/home/winter/code/TreeSeg"

# -------------------------- preprocess --------------------------
# TODO: 51
python3 -W ignore ../split_infer_merge_pipeline.py \
    --task preprocess_train \
    --tif_dir $data_root/tif \
    --area_polygon_dir $data_root/shp \
    --area_range list \
    --area_idx 52 53 54 55 56 57 \
    --interm_png_dir $data_root/interm_png