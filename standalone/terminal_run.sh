# !!!ALERT!!!: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

# !!!ALERT!!!: ndvi.tif contains -3.4028234663852886e+38, clip to range [0,1] right after read from tif

# data_root="/home/lenovo/treeseg-dataset/full_process"
data_root="/home/lenovo/treeseg-dataset/inference_train_data"
# data_root="/home/lenovo/treeseg-dataset/inference/all-views"
proj_root="/home/lenovo/code/TreeSeg"

# -------------------------- preprocess --------------------------
# python3 -W ignore split_infer_merge_pipeline.py \
#     --task preprocess_train \
#     --tif_dir $data_root/tif \
#     --area_polygon_dir $data_root/shp \
#     --area_range  0-2 \
#     --interm_png_dir $data_root/interm_png_nonorm


# -------------------------- split train --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_256 \
#     --split_unit 256 \
#     --norm_mode after_split

# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png_nonorm \
#     --sample_dir $data_root/sample_128_afternorm \
#     --split_unit 128 \
#     --norm_mode after_split

# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_108 \
#     --split_unit 108 \
#     --norm_mode after_split

# -------------------------- train --------------------------
# python3 torch_train.py \
#     --dataset_dir $data_root/... \
#     --model_dir $proj_root/...

# -------------------------- split inference --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_256 \
#     --split_unit 256

# python3 split_infer_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_108 \
#     --split_unit 108

# python3 split_infer_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_128_nonorm \
#     --split_unit 128 \
#     --norm_mode no_norm

# -------------------------- inference --------------------------
python3 split_infer_merge_pipeline.py \
    --task inference \
    --model_path $proj_root/checkpoints/nonorm.pth \
    --sample_dir $data_root/inference_sample_128_nonorm \
    --result_dir $data_root/torch_result/inference_result_128_nonorm

# -------------------------- merge --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task merge \
#     --input_dir $data_root/inference_sample_128_nonorm \
#     --result_dir $data_root/inference_result_128_nonorm \
#     --split_unit 128 \
#     --merge_dir $data_root/merge_result_128_nonorm \
#     --origin_tif $data_root/tif/pan-0.tif