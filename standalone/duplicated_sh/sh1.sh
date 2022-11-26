# ALERT: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

data_root="/home/lenovo/treeseg-dataset/full_process"

# -------------------------- preprocess --------------------------
python3 -W ignore split_merge_pipeline.py \
    --task preprocess_train \
    --tif_dir $data_root/tif \
    --area_polygon_dir $data_root/shp \
    --area_range 0-12 \
    --interm_png_dir $data_root/interm_png
