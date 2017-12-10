from geojson_to_pixel_arr import geojson_to_pixel_arr
from plot_truth_coords import plot_truth_coords
from create_building_mask import create_building_mask
from create_dist_map import create_dist_map
from plot_dist_transform import plot_dist_transform

import matplotlib.pyplot as plt
import numpy as np

import os
import glob

data_dir = "/media/hdd/Data/AWS/AOI 1 - Rio de Janeiro/processedBuildingLabels/samples/"
images = glob.glob(os.path.join(data_dir,"*.tif"))
json_files = glob.glob(os.path.join(data_dir,"*.geojson"))

img = images[0]
geojson = json_files[0]

pixel_coords, latlon_coords = geojson_to_pixel_arr(img,geojson)
# print(latlon_coords)

input_image = plt.imread(img)
plot_truth_coords(input_image, pixel_coords)

output_mask = "output/mask.tif"
create_building_mask(img,geojson,output_mask)

mask = plt.imread(output_mask)
plt.imshow(mask, cmap='bwr')
plt.show()

dist_map = "output/dist.npy"
create_dist_map(img, geojson, dist_map)
dist_image = np.load(dist_map)
plot_dist_transform(input_image, pixel_coords, dist_image)
