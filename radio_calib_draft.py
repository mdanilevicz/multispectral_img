#! /usr/bin/env python


'''
    This is based on the micasense radiometric calibration tutorial
    available at https://github.com/micasense/imageprocessing
'''
def main(): 
    # Runs with conda Micasense environment
    # Import Specific libraries
    import micasense.imageset as imageset
    import micasense.capture as capture
    import micasense.imageutils as imageutils
    import micasense.plotutils as plotutils
    #from mapboxgl.utils import df_to_geojson, create_radius_stops, scale_between
    import exiftool
    import cv2
    
    # Import general libraries
    import os, glob
    import multiprocessing
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    import subprocess
    import argparse
    import datetime

    
    # Build parser library
    parser = argparse.ArgumentParser()
    my_parser = argparse.ArgumentParser(description='Radiometric Calibration script for Micasense Altum images')
    parser.add_argument('-p', '--panel', required=True, help='Full path to the reference panel directory. It is important that only good quality centralised panel images are used as reference')
    parser.add_argument('-i', '--imageset', required=True, help='Full path to the images folder')
    parser.add_argument('-o', '--output', required=True, help='Full output path where the calibrated images and files will be saved')
    parser.add_argument('-t', '--thumbnail', default=True)
    parser.add_argument('--dls', default=True)
    parser.add_argument('--overwrite', default=False)
    args = parser.parse_args()
    
    # Define initial settings
    useDLS = args.dls
    overwrite = args.overwrite
    generateThumbnails = args.thumbnail
    imagePath = args.imageset
    panelPath = args.panel
    outputPath = args.output + '/stacks'
    thumbnailPath = args.output + '/thumbnail'
    start = datetime.datetime.now()
    
    # Create Panel Imageset
    panelset = imageset.ImageSet.from_directory(panelPath)
    panelCap = panelset.captures
    irradiances = []
    for capture in panelCap:
        if capture.panel_albedo() is not None and not any(v is None for v in capture.panel_albedo()):
            panel_reflectance_by_band = capture.panel_albedo()
            panel_irradiance = capture.panel_irradiance(panel_reflectance_by_band)
            irradiances.append(panel_irradiance)
        img_type='reflectance'
    # Get the mean reflectance per band considering all panel images
    df_panel = pd.DataFrame(irradiances)
    mean_irradiance = df_panel.mean(axis=0)
    mean_irradiance = mean_irradiance.values.tolist()
    
    # Load the Imageset
    imgset = imageset.ImageSet.from_directory(imagePath)
    data,columns = imgset.as_nested_lists()
    df_img = pd.DataFrame.from_records(data,index='timestamp',columns=columns)
   # geojson_data = df_to_geojson(df_img,columns[3:], lat='latitude', lon='longitude')
    
    # Creating the output paths and geojson
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if generateThumbnails and not os.path.exists(thumbnailPath):
        os.makedirs(thumbnailPath)
   # with open(os.path.join(outputPath, 'imageset.json'),'w') as f:
   #     f.write(str(geojson_data))
    
    # Imageset transforms
    # Alignment settings
    match_index = 1 # Index of the band I will try to match all others
    max_alignment_iterations = 30 #increase max_iterations for better results, but longer runtimes
    warp_mode = cv2.MOTION_HOMOGRAPHY # for Altum images only use HOMOGRAPHY
    pyramid_levels =1 # for images with Rigrelatives, setting this to 0 or 1 may improve the alignment
    
    ## Find the warp_matrices for one of the images
    #chose a random middle of the flight capture (like 50?)
    
    matrice_sample = imgset.captures[4]
    warp_matrices, alignment_pairs = imageutils.align_capture(matrice_sample,
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels)
    print("Finished Aligning, warp matrices={}".format(warp_matrices))
    
    # save warp_matrices used for the imgset alignment
    with open(os.path.join(outputPath, 'warp_matrices.txt'),'w') as f:
        f.write(str(warp_matrices))
    
    # Effectively UNWARP, ALIGN, CROP EDGES and get REFLECTANCE
    for i,capture in enumerate(imgset.captures):
        try:
            irradiance = mean_irradiance+[0]
        except NameError:
            irradiance = None
        
        #Create the output file names and path    
        outputFilename = capture.uuid + '.tif'
        thumbnailFilename = capture.uuid+'.jpg'
        fullOutputPath = os.path.join(outputPath, outputFilename)
        fullThumbnailPath= os.path.join(thumbnailPath, thumbnailFilename)
            
        # Check the waters
        if (not os.path.exists(fullOutputPath)) or overwrite:
            if(len(capture.images) == len(imgset.captures[0].images)):
                # Unwarp and Align and get Reflectance
                capture.create_aligned_capture(irradiance_list=irradiance,warp_matrices=warp_matrices)
                # Save the output images
                capture.save_capture_as_stack(fullOutputPath)
                if generateThumbnails:
                    capture.save_capture_as_rgb(fullThumbnailPath)
        
        # Clean cached data            
        capture.clear_image_data()
    
    # Extract the metadata from the captures list and save to log.csv
    def decdeg2dms(dd):
        is_positive = dd >= 0
        dd = abs(dd)
        minutes,seconds = divmod(dd*3600,60)
        degrees, minutes = divmod(minutes,60)
        degrees = degrees if is_positive else -degrees
        return (degrees, minutes,seconds)
    #Build file header
    header = "SourceFile,\
    GPSDateStamp,GPSTimeStamp,\
    GPSLatitude,GpsLatitudeRef,\
    GPSLongitude,GPSLongitudeRef,\
    GPSAltitude,GPSAltitudeRef,\
    FocalLength,\
    XResolution,YResolution,ResolutionUnits\n"
    
    lines= [header]
    # get the info from each capture
    for capture in imgset.captures:
        #get lat, lon, alt and time
        outputFilename = capture.uuid+'.tif'
        fullOutputPath= os.path.join(outputPath,outputFilename)
        lat, lon, alt = capture.location()
        #write to csv in format:
        # IMG_0199_1.tif,"33 deg 32' 9.73"" N","111 deg 51' 1.41"" W",526 m Above Sea Level
        latdeg, latmin, latsec = decdeg2dms(lat)
        londeg, lonmin, lonsec = decdeg2dms(lon)
        latdir = 'North'
        if latdeg < 0:
            latdeg = -latdeg
            latdir = 'South'
        londir = 'East'
        if londeg < 0:
            londeg = -londeg
            londir = 'West'
        resolution = capture.images[0].focal_plane_resolution_px_per_mm
    
        linestr = '"{}",'.format(fullOutputPath)
        linestr += capture.utc_time().strftime("%Y:%m:%d,%H:%M:%S,")
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},'.format(int(latdeg),int(latmin),latsec,latdir[0],latdir)
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},{:.1f} m Above Sea Level,Above Sea Level,'.format(int(londeg),int(lonmin),lonsec,londir[0],londir,alt)
        linestr += '{}'.format(capture.images[0].focal_length)
        linestr += '{},{},mm'.format(resolution,resolution)
        linestr += '\n' # when writing in text mode, the write command will convert to os.linesep
        lines.append(linestr)
    
    # Save the CSV with each capture metadata
    fullCsvPath = os.path.join(outputPath,'log.csv')
    with open(fullCsvPath, 'w') as csvfile: #create CSV
        csvfile.writelines(lines)
    
    # overwrite the image metadata
    if os.environ.get('exiftoolpath') is not None:
        exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
    else:
        exiftool_cmd = 'exiftool'
            
    cmd = '{} -csv="{}" -overwrite_original "{}"'.format(exiftool_cmd, fullCsvPath, outputPath)
    subprocess.check_call(cmd, shell=True)
    
    end= datetime.datetime.now()
    print("Running the scrip took: {}".format(end-start))
    
if __name__ == "__main__":
    main()
