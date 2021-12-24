import netCDF4
from netCDF4 import num2date
import numpy as np
import os
import pandas as pd
from pathlib import Path


class Preprocessing:

    def __init__(self, path):
        self.path = path

    def get_data(self):
        # Open netCDF4 file
        f = netCDF4.Dataset(self.path)

        # Extract variable
        SST = f.variables['SST']
        # Get dimensions assuming 3D: time, latitude, longitude
        self.time_dim, self.lat_dim, self.lon_dim = SST.get_dims()
        time_var = f.variables[time_dim.name]
        times = num2date(time_var[:], time_var.units)
        self.latitudes = f.variables['lat_rho'][:]
        self.longitudes = f.variables['lon_rho'][:]
        return SST

    def data_to_image(self, sst, path="/gpfs_common/share02/rhe/nkpatel8/SST_imgs/"):
        # Write input images
        day = 0
        for itr in SST:
            curr_df = pd.DataFrame(itr)
            curr_img = curr_df.to_numpy()
            img = Image.fromarray(curr_img)
            img = img.convert('RGB')
            img.save(path + "day_" + str(day) + ".jpeg")
            day += 1

    def load_dataset(self, path="/gpfs_common/share02/rhe/nkpatel8/SST_imgs/", curr_shape=(280, 353, 3), resized_shape=(512,512)):
        # Load dataset
        input_data = []
        image_id = 0
        ht, wd, cc = curr_shape
        ww, hh = resized_shape
        color = (255, 255, 255)
        xx = 0
        yy = 0
        data_dir = Path(path)
        for i in range(0, 60):
            img = np.array(Image.open(data_dir / f"day_{image_id}.jpeg"))
            resized = np.full((hh, ww, cc), color, dtype=np.float64)
            resized[yy:yy + ht, xx:xx + wd] = img
            input_data.append(resized)
            image_id += 1
        return input_data


