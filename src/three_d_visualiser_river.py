from collections import defaultdict
import heapq

from osgeo import gdal 

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import pickle

import pybullet as p
import pybullet_data
import time

def three_d_viz():
    global path
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    
    #+-+-+-+-+-+-+-+-+-+-+-+- Open the dtm as a raster image +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    DATASETS_PATH = "datasets"
    
    dtm_path_tiff = f"{DATASETS_PATH}/HiRISE/river.tiff"
    # dtm_path_tiff = f"{DATASETS_PATH}/HiRISE/utopia_planitia.tiff"
    

    dtm_image = gdal.Open(dtm_path_tiff)
    
    band1 = dtm_image.GetRasterBand(1) # Red channel 
    band2 = dtm_image.GetRasterBand(1) # Green channel 
    band3 = dtm_image.GetRasterBand(1) # Blue channel
    
    b1 = band1.ReadAsArray() 
    b2 = band2.ReadAsArray() 
    b3 = band3.ReadAsArray() 
    
    dtm_image_array = np.array(dtm_image)
    
    print(np.max(dtm_image_array))


    img = np.dstack((b1, b2, b3)) 

    min_val = -2739.01
    max_val = -2155.44
    # min_val = -3716.16
    # max_val = -3659.73
    
    img_clipped = np.clip(img, min_val, max_val)
    
    img_normalized = ((img_clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    x_start, y_start = 2100, 2100
    x_end, y_end = 2800, 2800
    # x_start, y_start = 1000, 1000
    # x_end, y_end = 1500, 1500
    
    print(img_normalized.shape)
    
    img_cropped = img_normalized[y_start:y_end, x_start:x_end]
    
    # print(img_cropped[200][300])
    
    plt.imshow(img_cropped)
    # plt.savefig('Cropped_Tiff.png')
    plt.show()
    
    img_cropped = img_cropped[:, :, :1]
    
    heightmap_data = (img_cropped / img_cropped.max()).flatten()
    
    
    heightmap_data = heightmap_data[:heightmap_data.shape[0]]
    
    print(heightmap_data.shape)
    
    dtm_size = img_cropped.shape[0]
    # heightmap_data = np.zeros(dtm_size * dtm_size)
    
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.01, 0.01, 2.5],  # Scale x, y, z
        # meshScale=[1, 1, 100],  # Scale x, y, z
        heightfieldTextureScaling=(dtm_size - 1) / 2,
        heightfieldData=heightmap_data,
        numHeightfieldRows=dtm_size,
        numHeightfieldColumns=dtm_size
    )
    print("Here 1")
    
    terrain_id = p.createMultiBody(0, terrain_shape, basePosition=[0, 0, 0.1], baseOrientation=[0, 0, -0.707, 0.707])
    print("Here 2")
    
    robot_start_pos = [0.1, 0.1, 0.2]
    print("Here 3")
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Initial facing direction
    print("Here 4")
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orientation, globalScaling=0.1)
    print("Here 5")
    print("Loaded urdf")
    
    while True:
        p.stepSimulation()
        time.sleep(1./240.)


if __name__ == "__main__":
    three_d_viz()
    
    