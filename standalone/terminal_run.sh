# ALERT: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

data_root="/home/lenovo/treeseg-dataset/preprocess/1014/predict"

# python3 split_merge_pipeline.py \
#    --task preprocess_train \
#    --tif_dir $data_root/tif \
#    --area_polygon_dir $data_root/shp \
#    --area_range 3-4 \
#    --interm_png_dir $data_root/interm_png
#
# python3 split_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_256 \
#     --split_unit 256 \
#     --norm_mode after_split

# python3 split_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_128 \
#     --split_unit 128 \
#     --norm_mode after_split

# python3 split_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_108 \
#     --split_unit 108 \
#     --norm_mode after_split

python3 split_merge_pipeline.py \
    --task split_inference \
    --tif_dir $data_root/tif \
    --sample_dir $data_root/inference_sample_256 \
    --split_unit 256

python3 split_merge_pipeline.py \
    --task split_inference \
    --tif_dir $data_root/tif \
    --sample_dir $data_root/inference_sample_108 \
    --split_unit 108

# python3 split_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_160 \
#     --split_unit 160