# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:36:07 2021

@author: portierl4527
"""

import os
import json
import glob
from pascal_voc_writer import Writer
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy
from tqdm import tqdm



#set classes
classes = {'traffic sign': 'traffic sign', 'traffic light':'traffic light'}
classes_keys = list(classes.keys())

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

#function to transform polygons to bounding boxes
def polygon_to_bbox(polygon):
    x_coordinates, y_coordinates = zip(*polygon)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


def convert_json(file):
    
    #if no relevant objects found in the image,
    #don't save the xml for the image
    relevant_file = False
    
    data = []
    with open(file, 'r') as f:
        file_data = json.load(f)

        for object in file_data['objects']:
            label, polygon = object['label'], object['polygon']
            
            #process only if label found in voc
            if label in classes_keys:
                polygon = np.array([x for x in polygon])
                bbox = polygon_to_bbox(polygon)
                data.append([classes[label]]+bbox)

        #if relevant objects found in image, set the flag to True
        if data:
            relevant_file = True

    return data, relevant_file

#save as an xml
def save_xml(img_path, img_shape, data, save_path):
    writer = Writer(img_path,img_shape[0], img_shape[1])
    for element in data:
        writer.addObject(element[0],element[1],element[2],element[3],element[4])
    writer.save(save_path)
    
    
valid_files = []
trainval_files = []
test_files = []

#make Annotations target directory if already doesn't exist


if __name__ == 'main':
    
    #set directories
    cityscapes_dir = '../datasets/cityscapes/cityscapes/'
    save_path = './cityscapes_voc_annotations/'

    cityscapes_dir_gt = os.path.join(cityscapes_dir, 'gtFine')
    
    ann_dir = os.path.join(save_path, 'VOC2007','Annotations')
    make_dir(ann_dir)
    
    
    
    
    
    for category in tqdm(os.listdir(cityscapes_dir_gt)):
        
        #no GT for test data
        if category == 'test': continue
        
        for city in os.listdir(os.path.join(cityscapes_dir_gt, category)):
    
            #read files
            files = glob.glob(os.path.join(cityscapes_dir, 'gtFine', category, city)+'/*.json')
            
            #process json files
            for file in files:
                data, relevant_file = convert_json(file)
                
                if relevant_file:
                    base_filename = os.path.basename(file)[:-21]
                    xml_filepath = os.path.join(ann_dir,base_filename + '_leftImg8bit.xml')
                    img_name = base_filename+'_leftImg8bit.png'
                    img_path = os.path.join(cityscapes_dir, 'leftImg8bit', category, city, base_filename+'_leftImg8bit.png')
                    img_shape = plt.imread(img_path).shape
                    valid_files.append([img_path, img_name])
                    
                    #make list of trainval and test files for voc format 
                    #lists will be stored in txt files
                    trainval_files.append(img_name[:-4]) if category == 'train' else test_files.append(img_name[:-4])
                    
                    #save xml file
                    save_xml(img_path, img_shape, data, xml_filepath)
                    
    images_savepath = os.path.join(save_path, 'VOC2007', 'JPEGImages')
    make_dir(images_savepath)


    for file in tqdm(valid_files):
        copy(file[0], os.path.join(images_savepath, file[1]))
    
