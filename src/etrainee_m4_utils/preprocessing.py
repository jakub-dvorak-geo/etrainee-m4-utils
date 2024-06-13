# etrainee_m4_utils/preprocessing.py

"""Preprocess imagery in memory for use in convolutional neural nets.

The module contains the following functions:

- `add(a, b)` - Returns the sum of two numbers.
- `subtract(a, b)` - Returns the difference of two numbers.
- `multiply(a, b)` - Returns the product of two numbers.
- `divide(a, b)` - Returns the quotient of two numbers.
"""

import numpy as np
import rasterio
from scipy.io import loadmat


class Image_tiler:
    """Tile imagery in RAM for use in convolutional neural nets."""

    def __init__(self, in_arr, out_shape=(256, 256),
                 out_overlap=128, offset=(0, 0)):
        """
        Initialize the class with required data.

        in_arr:         the numpy array to tile
        out_shape:      tuple of (height, width) of resulting tiles
        out_overlap:    int, number of pixels to overlap by
        offset:         tuple, offset from top left corner in pixels
        """
        self.in_arr = in_arr
        self.in_shape = in_arr.shape
        self.out_shape = out_shape
        self.out_overlap = out_overlap
        self.offset = offset

    def crop_image(self):
        """Crops the input image in order to be tileable."""
        height = self.out_shape[0] + self.offset[0]
        while True:
            height += (self.out_shape[0] - self.out_overlap)
            if self.in_shape[0] < height:
                height -= (self.out_shape[0] - self.out_overlap)
                break

        width = self.out_shape[1] + self.offset[1]
        while True:
            width += (self.out_shape[1] - self.out_overlap)
            if self.in_shape[1] < width:
                width -= (self.out_shape[1]-self.out_overlap)
                break

        self.crop_arr = self.in_arr[self.offset[0]:height,
                                    self.offset[1]:width, :]
        return self.crop_arr

    def tile_image(self):
        """Tiles the input image in order to use it in CNNs."""
        self.tiles_num_ver = int((self.in_shape[0] - self.out_shape[0])
                                 / (self.out_shape[0] - self.out_overlap)) + 1
        self.tiles_num_hor = int((self.in_shape[1] - self.out_shape[1])
                                 / (self.out_shape[1] - self.out_overlap)) + 1

        tiles_num = self.tiles_num_ver * self.tiles_num_hor
        self.tiles_arr = np.empty((tiles_num, self.out_shape[0],
                                  self.out_shape[1], self.in_shape[2]),
                                  self.in_arr.dtype)
        idx = 0

        for row in range(self.tiles_num_ver):
            for col in range(self.tiles_num_hor):
                row_start = row * (self.out_shape[0] - self.out_overlap)
                col_start = col * (self.out_shape[1] - self.out_overlap)
                self.tiles_arr[idx, :, :, :] = self.in_arr[
                    row_start:row_start+self.out_shape[0],
                    col_start:col_start+self.out_shape[1], :]
                idx += 1
        return self.tiles_arr
    
    def process_tiles(self, return_tile_dims=False):
        self.crop_image()
        self.tile_image()

        if return_tile_dims:
            tile_dims = {
                'tiles_num': (self.tiles_num_ver, self.tiles_num_hor),
                'cropped_shape': self.crop_arr.shape
            }
            return self.tiles_arr, tile_dims
        else:
            return self.tiles_arr


def normalize_tiles(in_dict, nodata_vals=[], is_training=False):
    """Normalize values between 0 and 1."""
    def normalize_input(arr):
        # normalize all values in array
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
        return arr_norm
    arr_dict = {}
    arr_dict['imagery'] = in_dict['imagery'].transpose([0, 3, 1, 2])
    for nodata_val in nodata_vals:
        arr_dict['imagery'][arr_dict['imagery'] == nodata_val] = 0
    arr_dict['imagery'] = normalize_input(arr_dict['imagery']
                                          .astype(np.float32))
    if is_training:
        ref_newshape = in_dict['reference'].shape[:-1]
        arr_dict['reference'] = in_dict['reference'].reshape(ref_newshape)
        arr_dict['reference'] = arr_dict['reference'].astype(np.int64)
        unique, counts = np.unique(arr_dict['reference'], return_counts=True)
        return arr_dict, unique, counts
    else:
        return arr_dict


def normalize_tiles_1d(in_dict, nodata_vals=[], is_training=False):
    """Normalize values between 0 and 1."""
    arr_dict = {}
    if is_training:
        norm_dict, _, _ = normalize_tiles(in_dict, nodata_vals, is_training)
        arr_dict['imagery'] = norm_dict['imagery'][:, None, :, 0, 0]
        arr_dict['reference'] = norm_dict['reference'][:, 0, 0]
        arr_dict['reference'] = arr_dict['reference'] - 1
        unique, counts = np.unique(arr_dict['reference'], return_counts=True)
        return arr_dict, unique, counts
    else:
        norm_dict = normalize_tiles(in_dict, nodata_vals, is_training)
        arr_dict['imagery'] = norm_dict['imagery'][:, None, :, 0, 0]
        return arr_dict


def normalize_tiles_3d(in_dict, nodata_vals=[], is_training=False):
    """Normalize values between 0 and 1."""
    arr_dict = {}
    if is_training:
        norm_dict, _, _ = normalize_tiles(in_dict, nodata_vals, is_training)
        arr_dict['imagery'] = norm_dict['imagery'][:, None, :, :, :]
        arr_dict['reference'] = norm_dict['reference']
        unique, counts = np.unique(arr_dict['reference'], return_counts=True)
        return arr_dict, unique, counts
    else:
        norm_dict = normalize_tiles(in_dict, nodata_vals, is_training)
        arr_dict['imagery'] = norm_dict['imagery'][:, None, :, :, :]
        return arr_dict


def read_rasterio(img_path: str, ref_path: str = None,
                  offset: tuple = (0, 0)) -> dict:
    """Read imagery for training or inference from any rasters compatible with\
    rasterio/GDAL. When used for training, provide a ref_path.

    Args:
        img_path: A path to the input file containing imagery.
        ref_path: A path to the input file containing reference.
        offset: [NOT IMPLEMENTED] Tuple of two values for movement in \
            height and width.

    Returns:
        A dictionary containing imagery, (reference), crs and transform.
    """
    # Create dict to collect loaded values
    out_dict: dict = {}

    if offset != (0, 0):
        raise NotImplementedError('Offseting is not currently implemented.')

    # Load the input imagery and geoinformation
    with rasterio.open(img_path) as img:
        out_dict['imagery'] = np.moveaxis(img.read(), 0, -1)
        out_dict['crs'] = img.crs
        out_dict['transform'] = img.transform

    if ref_path:
        with rasterio.open(ref_path) as ref:
            if out_dict['crs'] != ref.crs:
                raise Exception('The input rasters are in a different CRS.')
            elif out_dict['transform'] != ref.transform:
                raise Exception('The input rasters do not overlap.')
            else:
                out_dict['reference'] = np.moveaxis(ref.read(), 0, -1)

    return out_dict


def read_pavia_centre(img_path: str, ref_path: str = None,
                      out_shape: tuple = (1096, 1096, 102)) -> dict:
    """Read imagery for training or inference from matlab matrices of the \
    Pavia City Centre benchmark dataset, other datasets will not work. \
    Removes the gap in the middle of the original dataset.

    Args:
        img_path: A path to the input .mat file containing imagery.
        ref_path: A path to the input .mat file containing reference.
        out_shape: Tuple of three values (h, w, b), which are used as the \
        shape of output imagery raster. Needs to be compatible with selected \
        size of tiles.

    Returns:
        A dictionary containing imagery, (reference), crs and transform.
    """
    out_dict = {}

    raster_orig = loadmat(img_path)
    raster_orig_arr = raster_orig['pavia']
    # removes gap in the input dataset.
    imagery: np.ndarray = np.zeros(out_shape, dtype=np.uint16)
    imagery[:, :223, :] = raster_orig_arr[:out_shape[0], :223, :out_shape[2]]
    imagery[:, 605-(1096-out_shape[1]):, :] = raster_orig_arr[
        :out_shape[0], 224:, :out_shape[2]]
    out_dict['imagery'] = imagery

    # Add the reference dataset when used for training
    if ref_path:
        raster_gt = loadmat(ref_path)
        raster_gt_arr = raster_gt['pavia_gt'][:, :, None]
        reference: np.ndarray = np.zeros([out_shape[0], out_shape[1], 1],
                                         dtype=np.uint8)
        reference[:, :223, :] = raster_gt_arr[:out_shape[0], :223, :]
        reference[:, 605-(1096-out_shape[1]):, :] = raster_gt_arr[
            :out_shape[0], 224:, :]
        out_dict['reference'] = reference

    # Add empty geolocation values for compatibility reasons
    out_dict['crs'] = None
    out_dict['transform'] = None
    return out_dict


def split_into_tiles(in_data: dict, tile_shape: tuple[int] = (256, 256),
                     tile_overlap: int = 128,
                     offset: tuple[int] = (0, 0)) -> dict:
    """Splits the imagery into tiles for training/imference. Necessary for
    memory and training reasons.

    Args:
        in_data: Dictionary containing the relevant data arrays.
        tile_shape: Tuple of two values (h, w), which are used as the \
            shape of individual tiles.
        tile_overlap: By how many pixels do individual tiles overlap.
        offset: [NOT IMPLEMENTED] Tuple of two values for movement in \
            height and width.

    Returns:
        A dictionary containing tiled imagery, (reference), crs and transform.
    """
    # copy crs and transform directly from input to output
    out_dict: dict = {'crs': in_data['crs'], 'transform': in_data['transform']}

    # Raise error if trying to use offset
    if offset != (0, 0):
        raise NotImplementedError('Offseting is not currently implemented.')

    # Tile imagery in the input dataset
    img_tile_processor = Image_tiler(in_data['imagery'], tile_shape,
                                     tile_overlap, offset)
    tile_arr, tile_dims = img_tile_processor.process_tiles(True)
    out_dict['imagery'] = tile_arr
    out_dict.update(tile_dims)

    # If the input dataset contains reference data, tile it too
    if in_data['reference'] is not None:
        ref_tile_processor = Image_tiler(in_data['reference'], tile_shape,
                                         tile_overlap, offset)
        out_dict['reference'] = ref_tile_processor.process_tiles(False)
    return out_dict


def remove_nodata_tiles(in_data: dict, nodata_val: int = 0,
                        min_area: float = 1.) -> dict:
    """Removes tiles without any reference pixels.

    Args:
        in_data: Dictionary containing the relevant tile arrays.
        nodata_ref: Nodata value in the reference raster.
        min_area: [NOT IMPLEMENTED]

    Returns:
        A dictionary containing tiled imagery, (reference), crs and transform.
    """

    if min_area != 1.0:
        raise NotImplementedError('Minimal area is not currently implemented.')

    out_dict: dict = {key: in_data[key] for key in
                      ['crs', 'transform', 'tiles_num', 'cropped_shape']}

    mask: np.ndarray = np.any(in_data['reference'] != nodata_val,
                              axis=(1, 2, 3))
    # Use the mask to filter the array
    out_dict['imagery'] = in_data['imagery'][mask]
    out_dict['reference'] = in_data['reference'][mask]

    return out_dict


def main():
    imagery_path = 'E:/datasets/etrainee/BL_202008_imagery.tif'
    reference_path = 'E:/datasets/etrainee/BL_202008_reference.tif'

    def _test_rasterio(img_path, ref_path):
        loaded_img = read_rasterio(img_path, ref_path)
        print('-------------------------------------------')
        print('Test rasterio')
        print(loaded_img.keys())
        print(loaded_img['imagery'].shape)
        print(loaded_img['reference'].shape)
        return loaded_img

    def _test_split_into_tiles(img_path, ref_path, t_shp, t_overlap):
        loaded_raster = _test_rasterio(img_path, ref_path)
        tiles = split_into_tiles(loaded_raster, t_shp, t_overlap)
        print('-------------------------------------------')
        print('Test split into tiles')
        print(tiles.keys())
        print(tiles['imagery'].shape)
        print(tiles['reference'].shape)
        return tiles

    def _test_remove_nodata(img_path, ref_path, t_shp, t_overlap):
        tiles = _test_split_into_tiles(img_path, ref_path, t_shp, t_overlap)
        filtered = remove_nodata_tiles(tiles, nodata_val=0)
        print('-------------------------------------------')
        print('Test remove nodata tiles')
        print(filtered.keys())
        print(filtered['imagery'].shape)
        print(filtered['reference'].shape)
        return filtered

    tile_shape = (64, 64)
    tile_overlap = 32
    # _test_rasterio(imagery_path, reference_path)
    # _test_split_into_tiles(imagery_path, reference_path, tile_shape, tile_overlap)
    _test_remove_nodata(imagery_path, reference_path, tile_shape, tile_overlap)


if __name__ == '__main__':
    main()
"""
    dummy_filename = 'C:\\Users\\dd\\Pictures\\DSC_0084.jpg'
    dummy_arr = imageio.imread(dummy_filename)  # .astype(np.float32)

    print(dummy_arr.dtype)
    print(dummy_arr.shape)

    dummy_dataset = Image_tiler(dummy_arr, (256, 256), 128, (0, 0))
    dummy_crop = dummy_dataset.crop_image()
    print(dummy_crop.shape)
    dummy_tiles = dummy_dataset.tile_image()
    print(dummy_tiles.shape)
"""
