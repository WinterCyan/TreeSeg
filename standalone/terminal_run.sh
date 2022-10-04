# ALERT: modify geopandas lib (base.py intersection() as follows: )
# try: 
#     return _binary_geo("intersection", self, other)
# except Exception:
#     return []

python3 split_merge_pipeline.py \
    --task preprocess_train \
    --tif_dir /Users/wintercyan/LocalDocuments/treeseg-resource/full_area2/data \
    --area_polygon_dir /Users/wintercyan/LocalDocuments/treeseg-resource/full_area2/data \
    --area_range 2-3 \
    --interm_png_dir /Users/wintercyan/LocalDocuments/treeseg-resource/full_area2/interm_png

python3 split_merge_pipeline.py \
    --task split_train \
    --interm_png_dir /Users/wintercyan/LocalDocuments/treeseg-resource/full_area2/interm_png \
    --sample_dir /Users/wintercyan/LocalDocuments/treeseg-resource/full_area2/sample \
    --split_unit 256 \
    --norm_mode after_split
