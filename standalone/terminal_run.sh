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
train_data_root="/home/winter/code-resource/treeseg/trainingdata"
infer_data_root="/home/winter/code-resource/treeseg/inferdata"
# data_root="/home/lenovo/treeseg-dataset/inference/all-views"
proj_root="/home/winter/code/TreeSeg"

# -------------------------- preprocess --------------------------
# python3 -W ignore split_infer_merge_pipeline.py \
#     --task preprocess_train \
#     --tif_dir $data_root/tif \
#     --area_polygon_dir $data_root/shp \
#     --area_range list \
#     --area_idx 17 \
#     --interm_png_dir $data_root/interm_png


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
#     --sample_dir $data_root/trainsample_128_onsample \
#     --split_unit 128 \
#     --norm_mode on_sample

# -------------------------- train --------------------------
# python3 torch_train.py \
#     --dataset_dir $train_data_root/trainsample_128_onsample \
#     --model_dir $proj_root/checkpoints \
#     --model_name on_sample_thr001 \
#     --load $proj_root/checkpoints/on_sample_thr001_epoch60.pth \
#     --load_epoch 60

# -------------------------- split inference --------------------------
# python3 split_infer_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $infer_data_root/tif \
#     --sample_dir $infer_data_root/sample128 \
#     --split_unit 128

# python3 split_infer_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $data_root/tif \
#     --sample_dir $data_root/inference_sample_108 \
#     --split_unit 108

# python3 split_infer_merge_pipeline.py \
#     --task split_inference \
#     --tif_dir $infer_data_root/tif \
#     --sample_dir $infer_data_root/sample_128_onsample \
#     --split_unit 128 \
#     --norm_mode on_sample

# # -------------------------- inference --------------------------
python3 split_infer_merge_pipeline.py \
    --task inference \
    --model_path $proj_root/checkpoints/on_sample_thr001_epoch60.pth \
    --sample_dir $infer_data_root/sample_128_onsample \
    --result_dir $infer_data_root/result/128_onsample_thr001

# # -------------------------- merge --------------------------
python3 split_infer_merge_pipeline.py \
    --task merge \
    --input_dir $infer_data_root/sample_128_onsample \
    --result_dir $infer_data_root/result/128_onsample_thr001 \
    --split_unit 128 \
    --merge_dir $infer_data_root/result/merge_128_onsample_thr001 \
    --origin_tif_dir $infer_data_root/tif
