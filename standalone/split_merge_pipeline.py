"""
dataset preprocessing, split, (prediction), merge pipeline
1. input: pan.tif, ndvi.tif
2. tif -> read, normalization(BEFORE/AFTER split), write as png
3. split, assign index, save as datdaset
4. load dataset and inference
5. merge results, annotation corresponding to img
6. convert to shp, count tree
"""

import argparse
from ast import arg
import sys
sys.path.append("/home/winter/code/TreeSeg/notebooks/")
import numpy as np
import rasterio
import os
import rasterio.mask
import rasterio.warp
import rasterio.merge
import geopandas as gps
import pandas as pd
import shapely
from shapely.geometry import box
import json
import PIL.ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def image_normalize(im, axis = (0,1), c = 1e-8):
    return (im - im.mean(axis)) / (im.std(axis) + c)

def calculateBoundaryWeight(polygonsInArea, scale_polygon = 1.5, output_plot = True): 
    '''
    For each polygon, create a weighted boundary where the weights of shared/close boundaries is higher than weights of solitary boundaries.
    '''
    # If there are polygons in a area, the boundary polygons return an empty geo dataframe
    if not polygonsInArea:
        return gps.GeoDataFrame({})
    tempPolygonDf = pd.DataFrame(polygonsInArea)
    tempPolygonDf.reset_index(drop=True,inplace=True)
    tempPolygonDf = gps.GeoDataFrame(tempPolygonDf.drop(columns=['Id']))
    new_c = []
    #for each polygon in area scale, compare with other polygons:
    for i in tqdm(range(len(tempPolygonDf)), desc="process polygons"):
        pol1 = gps.GeoSeries(tempPolygonDf.iloc[i][0])
        sc = pol1.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
        scc = pd.DataFrame(columns=['id', 'geometry'])
        scc = scc.append({'id': None, 'geometry': sc[0]}, ignore_index=True)
        scc = gps.GeoDataFrame(pd.concat([scc]*len(tempPolygonDf), ignore_index=True))

        pol2 = gps.GeoDataFrame(tempPolygonDf[~tempPolygonDf.index.isin([i])])
        #scale pol2 also and then intersect, so in the end no need for scale
        pol2 = gps.GeoDataFrame(pol2.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center'))
        pol2.columns = ['geometry']

        ints = scc.intersection(pol2)
        for k in range(len(ints)):
            if ints.iloc[k]!=None:
                if ints.iloc[k].is_empty !=1:
                    new_c.append(ints.iloc[k])
    new_c = gps.GeoSeries(new_c)
    new_cc = gps.GeoDataFrame({'geometry': new_c})
    new_cc.columns = ['geometry']
    bounda = gps.overlay(new_cc, tempPolygonDf, how='difference')
    if output_plot:
        fig, ax = plt.subplots(figsize = (10,10))
        bounda.plot(ax=ax,color = 'red')
        plt.show()
    #change multipolygon to polygon
    bounda = bounda.explode()
    bounda.reset_index(drop=True,inplace=True)
    #bounda.to_file('boundary_ready_to_use.shp')
    return bounda

def dividePolygonsInTrainingAreas(trainingPolygon, trainingArea, print_polygon_num=False):
    '''
    Assign annotated ploygons in to the training areas.
    '''
    # For efficiency, assigned polygons are removed from the list, we make a copy here. 
    cpTrainingPolygon = trainingPolygon.copy()
    splitPolygons = {}
    polygon_num_in_area = []

    # ---------------------- counting polygons number ----------------------
    if print_polygon_num:
        for i in tqdm(trainingArea.index, desc="counting polygons in areas"):
            polygon_count = 0
            for j in cpTrainingPolygon.index:
                if trainingArea.loc[i]['geometry'].intersects(cpTrainingPolygon.loc[j]['geometry']):
                    polygon_count += 1
            polygon_num_in_area.append(polygon_count)
        print(f"polygons in area: {polygon_num_in_area}")
    # ---------------------- counting polygons number ----------------------

    for i in tqdm(trainingArea.index, desc="divide polygons in training areas"):
        spTemp = []
        allocated = []
        for j in cpTrainingPolygon.index:
            if trainingArea.loc[i]['geometry'].intersects(cpTrainingPolygon.loc[j]['geometry']):
                spTemp.append(cpTrainingPolygon.loc[j])
                allocated.append(j)

            # Order of bounds: minx miny maxx maxy
        boundary = calculateBoundaryWeight(spTemp, scale_polygon = 1.5, output_plot = False)
        splitPolygons[trainingArea.loc[i]['id']] = {'polygons':spTemp, 'boundaryWeight': boundary, 'bounds':list(trainingArea.bounds.loc[i]),}
        cpTrainingPolygon = cpTrainingPolygon.drop(allocated)
    return splitPolygons

def readInputImages(imageBaseDir, rawImageFileType, rawNdviImagePrefix, rawPanImagePrefix):
    """
    Reads all images with prefix ndvi_image_prefix and image_file_type datatype in the image_base_dir directory.
    """     
    
    ndviImageFn = []
    for root, dirs, files in os.walk(imageBaseDir):
        for file in files:
            if file.endswith(rawImageFileType) and file.startswith(rawNdviImagePrefix):
                 ndviImageFn.append(os.path.join(root, file))
    panImageFn = [fn.replace(rawNdviImagePrefix, rawPanImagePrefix) for fn in ndviImageFn]
    print(panImageFn)
    inputImages = list(zip(ndviImageFn,panImageFn))
    return inputImages

def drawPolygons(polygons, shape, outline, fill):
    """
    From the polygons, create a numpy mask with fill value in the foreground and 0 value in the background.
    Outline (i.e the edge of the polygon) can be assigned a separate value.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    #Syntax: PIL.ImageDraw.Draw.polygon(xy, fill=None, outline=None)
    #Parameters:
    #xy – Sequence of either 2-tuples like [(x, y), (x, y), …] or numeric values like [x, y, x, y, …].
    #outline – Color to use for the outline.
    #fill – Color to use for the fill.
    #Returns: An Image object.
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=outline, fill=fill)
    mask = np.array(mask)#, dtype=bool)   
    return(mask)

def rowColPolygons(areaDf, areaShape, profile, filename, outline, fill):
    """
    Convert polygons coordinates to image pixel coordinates, create annotation image using drawPolygons() and write the results into an image file.
    """
    transform = profile['transform']
    polygons = []
    for i in areaDf.index:
        gm = areaDf.loc[i]['geometry']
        if isinstance(gm, shapely.geometry.MultiPolygon):
            continue
        a,b = zip(*list(gm.exterior.coords))
        row, col = rasterio.transform.rowcol(transform, a, b)
        zipped = list(zip(row,col)) #[list(rc) for rc in list(zip(row,col))]
        polygons.append(zipped)
    with open(filename, 'w') as outfile:  
        json.dump({'Trees': polygons}, outfile)
    mask = drawPolygons(polygons,areaShape, outline=outline, fill=fill)    
    profile['dtype'] = rasterio.int16
    with rasterio.open(filename.replace('json', 'png'), 'w', **profile) as dst:
        dst.write(mask.astype(rasterio.int16), 1)

def writeExtractedImageAndAnnotation(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize=True):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    """
    try:
        for band, imFn in zip(bands, imagesFilename):
            # Rasterio reads file channel first, so the sm[0] has the shape [1 or ch_count, x,y]
            # If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
            dt = sm[0][band].astype(profile['dtype'])
            if normalize: # Note: If the raster contains None values, then you should normalize it separately by calculating the mean and std without those values.
                dt = image_normalize(dt, axis=None) #  Normalize the image along the width and height, and since here we only have one channel we pass axis as None
            with rasterio.open(os.path.join(writePath, imFn+'-area{}.png'.format(writeCounter)), 'w', **profile) as dst:
                    dst.write(dt, 1) 
        if annotationFilename:
            annotation_json_filepath = os.path.join(writePath,annotationFilename+'-area{}.json'.format(writeCounter))
            # The object is given a value of 1, the outline or the border of the object is given a value of 0 and rest of the image/background is given a a value of 0
            rowColPolygons(polygonsInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, annotation_json_filepath, outline=0, fill = 1)
        if boundaryFilename:
            boundary_json_filepath = os.path.join(writePath,boundaryFilename+'-area{}.json'.format(writeCounter))
            # The boundaries are given a value of 1, the outline or the border of the boundaries is also given a value of 1 and rest is given a value of 0
            rowColPolygons(boundariesInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, boundary_json_filepath, outline=1 , fill=1)
        return(writeCounter+1)
    except Exception as e:
        print(e)
        print("Something nasty happened, could not write the annotation or the mask file!")
        return writeCounter
        
def findOverlap(img, areasWithPolygons, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter=1):
    """
    Finds overlap of image with a training area.
    Use writeExtractedImageAndAnnotation() to write the overlapping training area and corresponding polygons in separate image files.
    """
    overlapppedAreas = set()
    for areaID, areaInfo in areasWithPolygons.items():
        #Convert the polygons in the area in a dataframe and get the bounds of the area. 
        polygonsInAreaDf = gps.GeoDataFrame(areaInfo['polygons'])
        boundariesInAreaDf = gps.GeoDataFrame(areaInfo['boundaryWeight'])    
        bboxArea = box(*areaInfo['bounds'])
        bboxImg = box(*img.bounds)
        #Extract the window if area is in the image
        if(bboxArea.intersects(bboxImg)):
            profile = img.profile  
            sm = rasterio.mask.mask(img, [bboxArea], all_touched=True, crop=True )
            profile['height'] = sm[0].shape[1]
            profile['width'] = sm[0].shape[2]
            profile['transform'] = sm[1]
            # That's a problem with rasterio, if the height and the width are less then 256 it throws: ValueError: blockysize exceeds raster height 
            # So I set the blockxsize and blockysize to prevent this problem
            profile['blockxsize'] = 32
            profile['blockysize'] = 32
            profile['count'] = 1
            profile['dtype'] = rasterio.float32
            # writeExtractedImageAndAnnotation writes the image, annotation and boundaries and returns the counter of the next file to write. 
            writeCounter = writeExtractedImageAndAnnotation(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter)
            overlapppedAreas.add(areaID)
    return(writeCounter, overlapppedAreas)

def extractAreasThatOverlapWithTrainingData(inputImages, areasWithPolygons, writePath, ndviFilename, panFilename, annotationFilename, boundaryFilename, bands, writeCounter):
    """
    Iterates over raw ndvi and pan images and using findOverlap() extract areas that overlap with training data. The overlapping areas in raw images are written in a separate file, and annotation and boundary file are created from polygons in the overlapping areas.
    Note that the intersection with the training areas is performed independently for raw ndvi and pan images. This is not an ideal solution and it can be combined in the future.
    """
    if not os.path.exists(writePath):
        os.makedirs(writePath)
        
    overlapppedAreas = set()                   
    for imgs in tqdm(inputImages, desc="read tif images"):
        ndviImg = rasterio.open(imgs[0])
        panImg = rasterio.open(imgs[1])

        ncpan, imOverlapppedAreasPan = findOverlap(panImg, areasWithPolygons, writePath=writePath, imageFilename=[panFilename], annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, writeCounter=writeCounter )
        ncndvi,imOverlapppedAreasNdvi = findOverlap(ndviImg, areasWithPolygons, writePath=writePath, imageFilename=[ndviFilename], annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, writeCounter=writeCounter)
        if ncndvi == ncpan:
            writeCounter = ncndvi
        else:
            print('Couldnt create mask!!!')
            print(ncndvi)
            print(ncpan)
            break;
        if overlapppedAreas.intersection(imOverlapppedAreasNdvi):
            print(f'Information: Training area(s) {overlapppedAreas.intersection(imOverlapppedAreasNdvi)} spans over multiple raw images. This is common and expected in many cases. A part was found to overlap with current input image.')
        overlapppedAreas.update(imOverlapppedAreasNdvi)
    
    allAreas = set(areasWithPolygons.keys())
    if allAreas.difference(overlapppedAreas):
        print(f'Warning: Could not find a raw image correspoinding to {allAreas.difference(overlapppedAreas)} areas. Make sure that you have provided the correct paths!')

def split_inference_samples(tif_dir, sample_dir, split_unit, norm_mode="after"):
    """read tif & split into pngs

    Args:
        tif_dir: folder contains ALL pan-ndvi tif pairs
        sample_dir: folder contains ALL training/inference samples
        split_unit: pixel size of split square
        norm_mode: norm image before/after split
    """

    assert norm_mode == "before_split" or norm_mode == "after_split", "norm_mode argument NOT valid!"

    tif_dir = tif_dir.rstrip("/")
    sample_dir = sample_dir.rstrip("/")
    all_pan_files_names = [name for name in os.listdir(tif_dir) if (name.__contains__("pan") or name.__contains__("PAN")) and name.endswith(".tif")]
    all_ndvi_files_names = [name for name in os.listdir(tif_dir) if (name.__contains__("ndvi") or name.__contains__("NDVI")) and name.endswith(".tif")]
    for pan_item in all_pan_files_names:
        assert all_ndvi_files_names.count(pan_item.replace("pan","ndvi")) or all_ndvi_files_names.count(pan_item.replace("PAN","NDVI")), "cannot pair up all tif pan & ndvi images!"
    print(f"found {len(all_pan_files_names)} pairs of tif images in {tif_dir} ...")
    for pan_item in all_pan_files_names:
        pan_full_path = f"{tif_dir}/{pan_item}"
        ndvi_item = ""
        if pan_item.__contains__("pan"):
            ndvi_item = pan_item.replace("pan", "ndvi")
        elif pan_item.__contains__("PAN"):
            ndvi_item = pan_item.replace("PAN", "NDVI")
        assert ndvi_item != ""
        ndvi_full_path = f"{tif_dir}/{ndvi_item}"
        print(pan_full_path, ndvi_full_path)

        pan_dataset = rasterio.open(pan_full_path)
        pan_profile = pan_dataset.profile
        ndvi_dataset = rasterio.open(ndvi_full_path)
        ndvi_profile = ndvi_dataset.profile

        pan_profile['dtype'] = rasterio.float32
        pan_profile['height'] = split_unit
        pan_profile['width'] = split_unit
        pan_profile['blockxsize'] = 32
        pan_profile['blockysize'] = 32
        pan_profile['count'] = 1

        ndvi_profile['dtype'] = rasterio.float32
        ndvi_profile['height'] = split_unit
        ndvi_profile['width'] = split_unit
        ndvi_profile['blockxsize'] = 32
        ndvi_profile['blockysize'] = 32
        ndvi_profile['count'] = 1

        pan_arr = pan_dataset.read(1).astype(pan_profile['dtype'])
        ndvi_arr = ndvi_dataset.read(1).astype(ndvi_profile['dtype'])
        if norm_mode=="before_split":
            pan_arr = image_normalize(pan_arr)
            ndvi_arr = image_normalize(ndvi_arr)
        height, width = pan_arr.shape
        print(f"read tif, shape: {height}x{width}")

        split_rows = int(height/split_unit)
        split_cols = int(width/split_unit)
        print(f"spliting into {split_rows}x{split_cols} squares...")
        for r in range(split_rows):
            for c in range(split_cols):
                idx_str = f"r{r}c{c}"
                print(f"writing {idx_str}...")
                pan_sample = pan_arr[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                ndvi_sample = ndvi_arr[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                if norm_mode=="after_split":
                    pan_sample = image_normalize(pan_sample)
                    ndvi_sample = image_normalize(ndvi_sample)
                with rasterio.open(f"{sample_dir}/{idx_str}-{pan_item.replace('tif','png')}", 'w', **pan_profile) as dst:
                    dst.write(pan_sample, 1)
                    print(pan_sample.shape)
                    print(f"max: {np.max(pan_sample)}, min: {np.min(pan_sample)}")
                    dst.close()
                with rasterio.open(f"{sample_dir}/{idx_str}-{ndvi_item.replace('tif','png')}", 'w', **ndvi_profile) as dst:
                    dst.write(ndvi_sample, 1)
                    dst.close()

def preprocess_training_samples(tif_dir, area_polygon_dir, area_range, interm_png_dir, norm_mode="after_split"):
    """read tif, shp & split into training samples

    Args:
        tif_dir: folder contains tif pairs
        area_polygon_dir: folder contains area & polygon shp
            tif & area & polygon filename patter: [pan-***.tif, ndvi-***.tif, area-***.shp/sbx..., polygon-***.shp/sbx...]
        sample_dir: folder to save samples, [pan, ndvi, annotation, weight]
        split_unit: pixel size of sample
        norm_mode: norm image before/after split. Defaults to "after".
    """

    tif_dir = tif_dir.rstrip("/")
    area_polygon_dir = area_polygon_dir.rstrip("/")
    interm_png_dir = interm_png_dir.rstrip("/")

    if not os.path.exists(interm_png_dir):
        os.makedirs(interm_png_dir)

    assert norm_mode == "before_split" or norm_mode == "after_split", "norm_mode argument NOT valid!"

    # get tif pairs, for every pair of tif, find area & polygon shp.
    all_pan_filenames = [name for name in os.listdir(tif_dir) if name.startswith("pan-") and name.endswith(".tif")]
    all_ndvi_filenames = [name for name in os.listdir(tif_dir) if name.startswith("ndvi-") and name.endswith(".tif")]
    all_area_filenames = [name for name in os.listdir(area_polygon_dir) if name.startswith("area-") and name.endswith(".shp")]
    all_polygon_filenames = [name for name in os.listdir(area_polygon_dir) if name.startswith("polygon-") and name.endswith(".shp")]

    for pan_item in all_pan_filenames:
        assert all_ndvi_filenames.count(pan_item.replace("pan","ndvi")), "cannot pair up all pan with ndvi image!"
        assert all_area_filenames.count(pan_item.replace("pan","area").replace(".tif",".shp")), "cannot pair up all pan with area file!"
        assert all_polygon_filenames.count(pan_item.replace("pan","polygon").replace(".tif",".shp")), "cannot pair up all pan with polygon file!"

    for pan_item in all_pan_filenames:
        print(f"pan_item: {pan_item}")
        # pan_full_path = f"{tif_dir}/{pan_item}"
        # ndvi_full_path = f"{tif_dir}/{pan_item.replace('pan', 'ndvi')}"
        area_full_path = f"{area_polygon_dir}/{pan_item.replace('pan','area').replace('.tif','.shp')}"
        polygon_full_path = f"{area_polygon_dir}/{pan_item.replace('pan','polygon').replace('.tif','.shp')}"
        areas = gps.read_file(area_full_path)
        polygons = gps.read_file(polygon_full_path)
        print(f'read a total of {polygons.shape[0]} object polygons and {areas.shape[0]} training areas.')
        
        if areas.crs != polygons.crs:
            print("warning: area & polygon CRS dose not match!")
            target_crs = polygons.crs
            areas = areas.to_crs(target_crs)
        print(f"polygon crs: {polygons.crs}, area crs: {areas.crs}")
        assert polygons.crs == areas.crs

        areas['id'] = range(areas.shape[0])
        # ------------- for test -------------
        if area_range != "all":
            begin_idx = int(area_range.split("-")[0])
            end_idx = int(area_range.split("-")[1])
            areas = areas[begin_idx:end_idx][:]
        # ------------- for test -------------
        areas_with_polygons = dividePolygonsInTrainingAreas(polygons, areas)
        print(f'assigned training polygons in {len(areas_with_polygons)} training areas and created weighted boundaries for ploygons')

        input_imgs = readInputImages(tif_dir, ".tif", "ndvi-", "pan-")
        print(f'found a total of {len(input_imgs)} pair of raw images to process!')

        write_counter = 0
        extractAreasThatOverlapWithTrainingData(
            inputImages=input_imgs,
            areasWithPolygons=areas_with_polygons,
            writePath=interm_png_dir, 
            ndviFilename=f"{pan_item.replace('pan','ndvi').replace('.tif','')}",
            panFilename=f"{pan_item.replace('.tif','')}",
            annotationFilename=f"{pan_item.replace('pan','annotation').replace('.tif','')}",
            boundaryFilename=f"{pan_item.replace('pan','boundary').replace('.tif','')}",
            bands=[0],
            writeCounter=write_counter
        )

        print("matching polygon and annotation finished.")

def split_training_samples(interm_png_dir, sample_dir, split_unit, norm_mode="after_split"):
    # read extracted png dataset, [pan, ndvi, annotation, boundary]
    # [pan,ndvi] -> [area 0, 1, ...] -> [pan/ndvi/anno/bound 0, 1, ...] -> [pan/ndvi/anno/bound 0_r0c0, 0_r0c1, ...]
    # a pair of tif -> multi areas   ->  multi png group (1 <-> 1 area)  ->  rows x cols square (1 <-> png group)
    # split: pan-0.tif ...  --->  pan-0_0.png  ---> pan-0_0_r0c1.png
    # merge: patch up split margins with 0s

    interm_png_dir = interm_png_dir.rstrip("/")
    sample_dir = sample_dir.rstrip("/")
    assert norm_mode == "before_split" or norm_mode == "after_split", "norm_mode argument NOT valid!"

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    pan_file_names = [name.replace('.png','') for name in os.listdir(interm_png_dir) if name.startswith('pan-') and name.endswith('.png')]
    for pan_item in pan_file_names:
        print(f"spliting {pan_item} by {split_unit} pixel...")
        pan_dataset = rasterio.open(f"{interm_png_dir}/{pan_item}.png")
        pan_profile = pan_dataset.profile
        ndvi_dataset = rasterio.open(f"{interm_png_dir}/{pan_item.replace('pan','ndvi')}.png")
        ndvi_profile = pan_dataset.profile
        annotation_dataset = rasterio.open(f"{interm_png_dir}/{pan_item.replace('pan','annotation')}.png")
        annotation_profile = pan_dataset.profile
        boundary_dataset = rasterio.open(f"{interm_png_dir}/{pan_item.replace('pan','boundary')}.png")
        boundary_profile = pan_dataset.profile

        pan_profile['dtype'] = rasterio.float32
        pan_profile['height'] = split_unit
        pan_profile['width'] = split_unit
        pan_profile['blockxsize'] = 32
        pan_profile['blockysize'] = 32
        pan_profile['count'] = 1

        ndvi_profile['dtype'] = rasterio.float32
        ndvi_profile['height'] = split_unit
        ndvi_profile['width'] = split_unit
        ndvi_profile['blockxsize'] = 32
        ndvi_profile['blockysize'] = 32
        ndvi_profile['count'] = 1

        annotation_profile['dtype'] = rasterio.float32
        annotation_profile['height'] = split_unit
        annotation_profile['width'] = split_unit
        annotation_profile['blockxsize'] = 32
        annotation_profile['blockysize'] = 32
        annotation_profile['count'] = 1

        boundary_profile['dtype'] = rasterio.float32
        boundary_profile['height'] = split_unit
        boundary_profile['width'] = split_unit
        boundary_profile['blockxsize'] = 32
        boundary_profile['blockysize'] = 32
        boundary_profile['count'] = 1

        pan_img = pan_dataset.read(1).astype(pan_profile['dtype'])
        ndvi_img = ndvi_dataset.read(1).astype(ndvi_profile['dtype'])
        annotation_img = annotation_dataset.read(1).astype(annotation_profile['dtype'])
        boundary_img = boundary_dataset.read(1).astype(boundary_profile['dtype'])
        assert pan_img.shape == ndvi_img.shape == annotation_img.shape == boundary_img.shape

        if norm_mode=="before_split":
            pan_img = image_normalize(pan_img)
            ndvi_img = image_normalize(ndvi_img)

        height, width = pan_img.shape
        row_count = int(height/split_unit)
        col_count = int(width/split_unit)
        for r in range(row_count):
            for c in range(col_count):
                idx_str = f"r{r}c{c}"
                pan_sample = pan_img[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                ndvi_sample = ndvi_img[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                if norm_mode=="after_split":
                    pan_sample = image_normalize(pan_sample)
                    ndvi_sample = image_normalize(ndvi_sample)
                annotation_sample = annotation_img[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                boundary_sample = boundary_img[r*split_unit:(r+1)*split_unit, c*split_unit:(c+1)*split_unit]
                with rasterio.open(f"{sample_dir}/{pan_item}-{idx_str}.png", 'w', **pan_profile) as dst:
                    dst.write(pan_sample, 1)
                    dst.close()
                with rasterio.open(f"{sample_dir}/{pan_item.replace('pan','ndvi')}-{idx_str}.png", 'w', **ndvi_profile) as dst:
                    dst.write(ndvi_sample, 1)
                    dst.close()
                with rasterio.open(f"{sample_dir}/{pan_item.replace('pan','annotation')}-{idx_str}.png", 'w', **annotation_profile) as dst:
                    dst.write(annotation_sample, 1)
                    dst.close()
                with rasterio.open(f"{sample_dir}/{pan_item.replace('pan','boundary')}-{idx_str}.png", 'w', **boundary_profile) as dst:
                    dst.write(boundary_sample, 1)
                    dst.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tif_dir", type=str)
    parser.add_argument("--area_polygon_dir", type=str)
    parser.add_argument("--area_range", type=str)
    parser.add_argument("--interm_png_dir", type=str)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--split_unit", type=int)
    parser.add_argument("--norm_mode", type=str, default="after_split")
    args = parser.parse_args()


    # if args.task == "split_inference":
    #     assert args.tif_dir is not None, "please set tif_dir arg!"
    #     assert args.area_polygon_dir is not None, "please set area_polygon_dir arg!"
    #     assert args.interm_png_dir is not None, "please set interm_png_dir arg!"
    #     split_inference_samples(
    #             tif_dir=args.tif_dir,
    #             sample_dir=args.sample_dir,
    #             split_unit=args.split_unit,
    #             norm_mode=args.norm_mode
    #         )

    if args.task == "preprocess_train":
        assert args.tif_dir is not None, "please set tif_dir arg!"
        assert args.area_polygon_dir is not None, "please set area_polygon_dir arg!"
        assert args.area_range is not None, "please set area_range arg!"
        assert args.interm_png_dir is not None, "please set interm_png_dir arg!"
        preprocess_training_samples(
            tif_dir=args.tif_dir,
            area_polygon_dir=args.area_polygon_dir,
            area_range=args.area_range,
            interm_png_dir=args.interm_png_dir,
            norm_mode=args.norm_mode
        )

    if args.task == "split_train":
        assert args.interm_png_dir is not None, "please set interm_png_dir arg!"
        assert args.sample_dir is not None, "please set sample_dir arg!"
        assert args.split_unit is not None, "please set split_unit arg!"
        split_training_samples(
            interm_png_dir=args.interm_png_dir,
            sample_dir=args.sample_dir,
            split_unit=args.split_unit,
            norm_mode=args.norm_mode
        )

    # inference()
    # merge_results()
