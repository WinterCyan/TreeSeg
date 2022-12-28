# !!!ALERT!!!: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

# !!!ALERT!!!: ndvi.tif contains -3.4028234663852886e+38, clip to range [0,1] right after read from tif

# norm mode:
# TODO, 0. on_view: norm on view
# 1. on_area: norm on area
# 2. on_sample: norm on sample
# 3. no_norm: no norm

# data_root="/home/lenovo/treeseg-dataset/full_process"
# data_root="/home/lenovo/treeseg-dataset/inference_train_data"
data_root="/home/winter/code-resource/treeseg/trainingdata"
# data_root="/home/lenovo/treeseg-dataset/inference/all-views"
proj_root="/home/winter/code/TreeSeg"

# -------------------------- preprocess --------------------------
python3 -W ignore split_infer_merge_pipeline.py \
    --task preprocess_train \
    --tif_dir $data_root/tif \
    --area_polygon_dir $data_root/shp \
    --area_range list \
    --area_idx 17 \
    --interm_png_dir $data_root/interm_png


# -------------------------- split train --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_256 \
#     --split_unit 256 \
#     --norm_mode on_sample

# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/trainsample_128_onsample \
#     --split_unit 128 \
#     --norm_mode on_sample

# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/trainsample_128_onarea \
#     --split_unit 128 \
#     --norm_mode on_area

# python3 split_infer_merge_pipeline.py \
#     --task split_train \
#     --interm_png_dir $data_root/interm_png \
#     --sample_dir $data_root/sample_108 \
#     --split_unit 108 \
#     --norm_mode on_sample

# -------------------------- train --------------------------
# python3 torch_train.py \
#     --dataset_dir $data_root/trainsample_128_onsample \
#     --model_dir $proj_root/checkpoints \
#     --model_name onsample

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
#     --sample_dir $data_root/inference_sample_128_beforenorm \
#     --split_unit 128 \
#     --norm_mode on_area

# # -------------------------- inference --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task inference \
#     --model_path $proj_root/checkpoints/beforenorm.pth \
#     --sample_dir $data_root/inference_sample_128_beforenorm \
#     --result_dir $data_root/torch_result/inference_result_128_beforenorm

# # -------------------------- merge --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task merge \
#     --input_dir $data_root/inference_sample_128_beforenorm \
#     --result_dir $data_root/torch_result/inference_result_128_beforenorm \
#     --split_unit 128 \
#     --merge_dir $data_root/torch_result/merge_result_128_beforenorm \
#     --merge_tif \
#     --origin_tif $data_root/tif/pan-0.tif