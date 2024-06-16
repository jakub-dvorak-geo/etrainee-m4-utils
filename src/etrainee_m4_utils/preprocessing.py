# etrainee_m4_utils/preprocessing.py

"""Preprocess imagery in memory for use in convolutional neural nets.

The module contains the following functions:

- `read_rasterio(img_path, ref_path, offset)` - Reads rasterio rasters.
- `read_pavia_centre(img_path, ref_path, out_shape)` - Reads Pavia city centre
    dataset from matlab .mat files.
- `split_into_tiles(in_data, tile_shape, tile_overlap, offset)` - Splits
    the input rasters into tiles for training/inference.
- `remove_nodata_tiles(in_data, nodata_val, min_area)` - Remove tiles without
    valid reference classes.
"""

import numpy as np
import rasterio
from scipy.io import loadmat
# from sklearn.preprocessing import StandardScaler


class _Tile_processor:
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
                'num_tiles': (self.tiles_num_ver, self.tiles_num_hor),
                'shape_cropped': self.crop_arr.shape
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
    """Splits the imagery into tiles for training/inference. Necessary for
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
    # Copy unchanged values from the input dict
    out_dict: dict = {key: in_data[key] for key in
                      ['crs', 'transform', 'num_tiles', 'shape_cropped']
                      if key in in_data}

    # Raise error if trying to use offset
    if offset != (0, 0):
        raise NotImplementedError('Offseting is not currently implemented.')

    # Tile imagery in the input dataset
    img_tile_processor = _Tile_processor(in_data['imagery'], tile_shape,
                                         tile_overlap, offset)
    tile_arr, tile_dims = img_tile_processor.process_tiles(True)
    out_dict['imagery'] = tile_arr
    out_dict.update(tile_dims)

    # If the input dataset contains reference data, tile it too
    if 'reference' in in_data:
        ref_tile_processor = _Tile_processor(in_data['reference'], tile_shape,
                                             tile_overlap, offset)
        out_dict['reference'] = ref_tile_processor.process_tiles(False)
    return out_dict


def remove_nodata_tiles(in_data: dict, nodata_val: int = 0,
                        min_area: float = 1.) -> dict:
    """Removes tiles without any reference pixels.

    Args:
        in_data: Dictionary containing the relevant tile arrays.
        nodata_val: Nodata value in the reference raster.
        min_area: [NOT IMPLEMENTED]

    Returns:
        A dictionary containing tiled imagery, (reference), crs and transform.
    """
    if 'reference' in in_data:
        if min_area != 1.0:
            raise NotImplementedError('Minimal area is not currently \
                                      implemented.')
        # Copy unchanged values from the input dict
        out_dict: dict = {key: in_data[key] for key in
                          ['crs', 'transform', 'num_tiles', 'shape_cropped']
                          if key in in_data}

        # Create a mask... true if a tile contains other values than nodata_val
        mask: np.ndarray = np.any(in_data['reference'] != nodata_val,
                                  axis=(1, 2, 3))
        # Use the mask to filter the array
        out_dict['imagery'] = in_data['imagery'][mask]
        out_dict['reference'] = in_data['reference'][mask]

        return out_dict

    else:
        return in_data


'''
def _reclass_tiles_zero_one(in_data: dict, nodata_val: int = 65535) -> dict:
    """Normalize values in each tile between 0 and 1.

    Args:
        in_data: Dictionary containing the relevant tile arrays.
        nodata_val: Nodata value of the imagery raster.

    Returns:
        A dictionary containing tiled imagery, (reference), crs and transform.
    """
    def _norm_input(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())
    # Copy unchanged values from the input dict
    out_dict: dict = {key: in_data[key] for key in
                      ['crs', 'transform', 'num_tiles', 'shape_cropped']}

    transposed: np.ndarray = in_data['imagery'].transpose([0, 3, 1, 2])
    transposed[transposed == nodata_val] = 0
    out_dict['imagery'] = _norm_input(transposed.astype(np.float32))

    if 'reference' in in_data:
        newshape_ref: tuple = in_data['reference'].shape[:-1]
        reshaped_ref: np.ndarray = in_data['reference'].reshape(newshape_ref)
        out_dict['reference'] = reshaped_ref.astype(np.int64)
        unique, counts = np.unique(out_dict['reference'], return_counts=True)
        out_dict['unique'] = unique
        out_dict['counts'] = counts

    return out_dict


def _standard_scaler(in_data: dict, nodata_val: int = 65535) -> dict:
    """

    Args:
        in_data: Dictionary containing the relevant tile arrays.
        nodata_val: Nodata value of the imagery raster.

    Returns:
        A dictionary containing tiled imagery, (reference), crs and transform.
    """

    # StandardScaler()

    if 'reference' in in_data:
        out_dict['reference'] = in_data['reference']
    return out_dict
'''


def main():
    pass


if __name__ == '__main__':
    main()
