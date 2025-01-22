import shutil
import numpy as np
import mrcfile
from emalign.read_write import read_mrc
from rand_rots import rand_rots
from src.fastrotate3d import fastrotate3d
from src.reshift_vol import reshift_vol_int

TEST_PATH = "C:/Users/yuval/anaconda3/Projects/EMalign Project/Test Data/Test Cases/A/"

############################
# Test for volume alignment:
############################

# np.random.seed(1337)


def transform_map(map_id):
    print("--> reading map")
    vol = read_mrc(TEST_PATH + map_id + '.mrc')
    print("--> transforming map")
    R_true = rand_rots(1).reshape((3, 3))
    vol_c = np.copy(vol)
    vol_rotated = fastrotate3d(vol_c, R_true)
    vol_rotated = np.flip(vol_rotated, axis=2)
    vol_rotated = reshift_vol_int(vol_rotated, np.array([-5, 0, 0]))
    print("--> saving transformed map")
    save_transformed_map(vol_rotated, map_id)
    print("\n")


def save_transformed_map(transform_vol, map_id):
    vol_filename = TEST_PATH + map_id + '.mrc'
    t_vol_filename = TEST_PATH + map_id + "_query.mrc"

    # Copy vol2 to save header:
    shutil.copyfile(vol_filename, t_vol_filename)

    # Change and save:
    mrc_fh = mrcfile.open(t_vol_filename, mode='r+')
    mrc_fh.set_data(transform_vol.astype('float32').T)
    mrc_fh.set_volume()
    mrc_fh.update_header_stats()
    mrc_fh.close()


if __name__ == '__main__':
    map_IDs = ["2660", "19195", "35413"]
    for map_ID in map_IDs:
        print("Map: " + map_ID)
