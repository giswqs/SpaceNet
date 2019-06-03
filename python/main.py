from geojson_to_pixel_arr import geojson_to_pixel_arr
from plot_truth_coords import plot_truth_coords
from create_building_mask import create_building_mask
from create_dist_map import create_dist_map
from plot_dist_transform import plot_dist_transform

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal,ogr,osr

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta


import os
import glob
import time

def numpy_to_tif(arr, out_path, template):
    no_data = 0
    # First of all, gather some information from the template file
    data = gdal.Open(template)
    [cols, rows] = arr.shape
    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    # nodatav = 0 #data.GetNoDataValue()
    # Create the file, using the information from the template file
    outdriver = gdal.GetDriverByName("GTiff")
    # http://www.gdal.org/gdal_8h.html
    # GDT_Byte = 1, GDT_UInt16 = 2, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6,
    outdata   = outdriver.Create(str(out_path), rows, cols, 1, gdal.GDT_Float32)
    # Write the array to the file, which is the original array in this example
    outdata.GetRasterBand(1).WriteArray(arr)
    # Set a no data value if required
    outdata.GetRasterBand(1).SetNoDataValue(no_data)
    # Georeference the image
    outdata.SetGeoTransform(trans)
    # Write projection information
    outdata.SetProjection(proj)
    return arr


def find_img_json(img_list, json_dir):
    json_list = []

    for img in img_list:
        img_name = os.path.splitext(os.path.basename(img))[0]
        json_name = img_name.replace('3band', 'Geo') + '.geojson'
        json_path = os.path.join(json_dir, json_name)
        if os.path.exists(json_path):
            json_list.append(json_path)
            # print(img)
            # print(json_path)
    return json_list



def create_dist_tifs(img_list, json_list, out_dir, out_npy, nrows, ncols):

    N_imgs = len(img_list)
    print('Total number of images: ', str(N_imgs))

    arrays = []
    for i, img in enumerate(img_list):
        # print(i)
        img_name = os.path.splitext(os.path.basename(img))[0]
        dist_npy_name = img_name.replace('3band', 'dist') + '.npy'
        dist_tif_name = img_name.replace('3band', 'dist') + '.tif'
        print(str(i), ": ", dist_tif_name)
        dist_npy_path = os.path.join(out_dir, dist_npy_name)
        dist_tif_path = os.path.join(out_dir, dist_tif_name)
        # print(json_list[i])
        # print(dist_npy_path)
        dist_arr = create_dist_map(img,json_list[i])
        dist_arr = dist_arr[0:nrows, 0:ncols]
        # dist_arr.shape = dist_arr.shape + (1,)
        arrays.append(dist_arr)
        # print(dist_arr.shape)
        # tmp = np.load(dist_npy_path)
        numpy_to_tif(dist_arr,dist_tif_path, img)
        dist_arr = None
        # print(tmp.shape)
        # plt.imshow(dist_arr)
        # plt.show()
    data = np.array(arrays)
    data.shape = data.shape + (1,)  # convert 2d array to 3d
    print(data.shape)
    np.save(out_npy, data)
    return  arrays


def images_to_numpy(input_dir, out_file, nrows, ncols, nsamples):
    arrays = []
    filelist = glob.glob(os.path.join(input_dir, '*.tif'))
    filelist = filelist[0:nsamples]
    print("Total number of images: ", len(filelist))
    for i, file in enumerate(filelist):
        print(str(i), ': ', file)
        img = plt.imread(file)[0:nrows, 0:ncols, :]
        # print(img.shape)
        arrays.append(img)
    data = np.array(arrays)
    np.save(out_file, data)
    print(data.shape)
    return data


def train_model(X_train, y_train, batch_size, epochs):

    nrows = train_x.shape[1]
    ncols = train_x.shape[2]
    nbands = train_x.shape[3]
    input_shape = (nrows, ncols, nbands)

    # batch_size = 32
    # # num_classes = 6
    # epochs = 12

    model = Sequential()



    # model.add(Dense(13, kernel_initializer='normal', activation='relu', input_shape=input_shape))
    # model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    # # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.summary()

    # model.add(Dense(units=64, activation='relu', input_shape=input_shape))
    # model.add(Dense(units=10, activation='relu'))
    # model.add(Flatten())
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    # model.summary()
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    # model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nrows*ncols))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    model.summary()
    model.fit(X_train, y_train, batch_size, epochs)

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # # model.add(Dense(num_classes, activation='softmax'))
    # model.summary()
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])
    # model.fit(train_x, train_y,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1)
    #           # validation_data=(x_test, y_test))
    return 0


if __name__ == '__main__':

    start_time = time.time()

    data_dir = "/media/hdd/Data/AWS/AOI 1 - Rio de Janeiro/processedBuildingLabels/"
    json_dir = os.path.join(data_dir, 'vectordata/geojson/')

    out_dir = os.path.join(data_dir, "output")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sample_dir = os.path.join(data_dir, "3band")
    images = glob.glob(os.path.join(sample_dir,"*.tif"))


    train_dir = os.path.join(data_dir, 'train')
    train_x_path = os.path.join(train_dir, "train_x.npy")
    train_y_path = os.path.join(train_dir, "train_y.npy")

    nrows = 406
    ncols = 438
    nsamples = 1000
    images = images[0:nsamples]
    # for img in images:
    #     print(img)
    json_files = glob.glob(os.path.join(sample_dir,"*.geojson"))
    json_files = find_img_json(images, json_dir)

    # train_x = images_to_numpy(sample_dir, train_x_path, nrows, ncols, nsamples)
    # create_dist_tifs(images, json_files, out_dir, train_y_path, nrows, ncols)

    train_x = np.load(train_x_path)
    train_y = np.load(train_y_path)

    print(train_x.shape)
    print(train_y.shape)

    batch_size = 50
    epochs = 5
    train_model(train_x, train_y, batch_size, epochs)

    # print(train_x.shape)
    # train_x = np.load(train_y_path)
    # print(train_x.shape)
    # print(train_x)
    # train_y = images_to_numpy(out_dir, train_y_path)


    # img = images[0]
    # geojson = json_files[0]

    # pixel_coords, latlon_coords = geojson_to_pixel_arr(img,geojson)
    #
    # for pixels in pixel_coords:
    #     print(pixels)
    #     print("\n")
    #
    # for latlon in latlon_coords:
    #     print(latlon)
    #     print("\n")
    # print(pixel_coords)
    # print(latlon_coords)
    #
    # input_image = plt.imread(img)
    # plot_truth_coords(input_image, pixel_coords)
    #
    # output_mask = "output/mask.tif"
    # create_building_mask(img,geojson,output_mask)
    #
    # mask = plt.imread(output_mask)
    # plt.imshow(mask, cmap='bwr')
    # plt.show()
    #
    # dist_map = "output/dist.npy"
    # create_dist_map(img, geojson, dist_map)
    # dist_image = np.load(dist_map)
    # plot_dist_transform(input_image, pixel_coords, dist_image)

    end_time = time.time()
    print("Total run time: ", str(end_time-start_time))