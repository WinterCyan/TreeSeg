# !!!ALERT!!!: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

# !!!ALERT!!!: ndvi.tif contains -3.4028234663852886e+38, clip to range [0,1] right after read from tif

# data_root="/home/lenovo/treeseg-dataset/full_process"
# data_root="/home/lenovo/treeseg-dataset/inference/all-views"
data_root="/home/lenovo/treeseg-dataset/inference/all-views"

# -------------------------- preprocess --------------------------
# python3 -W ignore split_merge_pipeline.py \
#     --task preprocess_train \
#     --tif_dir $data_root/tif \
#     --area_polygon_dir $data_root/shp \
#     --area_range  \
#     --interm_png_dir $data_root/interm_png


# -------------------------- split train --------------------------
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

# -------------------------- split inference --------------------------
# python3 split_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_256 \
#     --split_unit 256

# python3 split_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_108 \
#     --split_unit 108

# python3 split_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_128_no_norm \
#     --split_unit 128 \
#     --norm_mode no_norm

# # -------------------------- inference --------------------------
# python3 split_merge_pipeline.py \
#     --task inference \
#     --model_path /home/lenovo/code/TreeSeg/notebooks/saved_models/UNet/model-1115.h5 \
#     --sample_dir $data_root/inference_sample_128 \
#     --result_dir $data_root/inference_result_128 \

# -------------------------- merge --------------------------
python3 split_merge_pipeline.py \
    --task merge \
    --input_dir $data_root/inference_sample_128 \
    --result_dir $data_root/inference_result_128 \
    --split_unit 128 \
    --merge_dir $data_root/merge_result_128 \
    --origin_tif $data_root/tif/pan0.tif