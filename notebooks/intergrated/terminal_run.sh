python3 split_merge_pipeline.py \
    --task preprocess_train \
    --tif_dir /home/winter/code-resources/treeseg/full_area2/tif \
    --area_polygon_dir /home/winter/code-resources/treeseg/full_area2/shp \
    --area_range 2-3 \
    --interm_png_dir /home/winter/code-resources/treeseg/full_area2/interm_png

python3 split_merge_pipeline.py \
    --task split_train \
    --interm_png_dir /home/winter/code-resources/treeseg/full_area2/interm_png \
    --sample_dir /home/winter/code-resources/treeseg/full_area2/sample_256 \
    --split_unit 256 \
    --norm_mode after_split

python3 split_merge_pipeline.py \
    --task split_train \
    --interm_png_dir /home/winter/code-resources/treeseg/full_area2/interm_png \
    --sample_dir /home/winter/code-resources/treeseg/full_area2/sample_128 \
    --split_unit 128 \
    --norm_mode after_split

python3 split_merge_pipeline.py \
    --task split_train \
    --interm_png_dir /home/winter/code-resources/treeseg/full_area2/interm_png \
    --sample_dir /home/winter/code-resources/treeseg/full_area2/sample_96 \
    --split_unit 96 \
    --norm_mode after_split
