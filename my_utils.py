def yolo2bbox (x,y,w,h):
    "transform yolo label to bbox"
    width = 2048 #image size
    height = 1024
    
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    
    return xmin, xmax, ymin, ymax


def get_gps(jsonfile):
    "read gps file and return latitude and longitude"
    import json 
    
    with open(jsonfile, 'r') as f:
        file_data = json.load(f)
        
        lat = file_data['gpsLatitude']
        lon = file_data['gpsLongitude']
        
        lat, lon
        
    return lat, lon

def get_cam_params(jsonfile):
    "read cam_file and return baseline and fx"
    import json
    
    with open(jsonfile, 'r') as f:
        file_data = json.load(f)


    #extrinsic and intrinsic parameters
    extrinsic = file_data['extrinsic']
    intrinsic = file_data['intrinsic']
   
    #baseline and focal point
    baseline = extrinsic['baseline']
    fx = intrinsic['fx']
    
    return baseline,fx


def disp2depth (image,fx ,baseline):
    "read disparity image and transform to metric depth"
    
    import cv2
    import numpy as np
    
    disp_image = cv2.imread(image, cv2.IMREAD_UNCHANGED).astype(np.float32)
    disp_image[disp_image > 0] = (disp_image[disp_image > 0] - 1) / 256
    disp_image[disp_image > 0] = fx*baseline/(disp_image[disp_image > 0])
    disp_image[disp_image == 0] = 475
    
    np.round(disp_image,2)
    
    return np.round(disp_image,2)

