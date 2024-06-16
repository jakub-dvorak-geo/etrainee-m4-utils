# etrainee_m4_utils/postprocessing.py

"""Infer/export classification results.

The module contains the following functions:

###### TODO Add correct functions and reformat as table
|Function          | Description        |
-----------------------------------------
| `add(a, b)`      | Returns the sum of two numbers. |
| `subtract(a, b)` | Returns the difference of two numbers. |
| `multiply(a, b)` | Returns the product of two numbers.  |
| `divide(a, b)`   | Returns the quotient of two numbers. |
"""

import numpy as np
import torch
import rasterio


# TODO Add type assertion to model
def classify_tiles(model, tiles_to_classify: np.ndarray,
                   dim: str = '2D') -> np.ndarray:
    """Classifies individual tiles using a PyTorch model.

    Args:
        model: The Pytorch model to use for classification
        tiles_to_classify: The classified array
        dim: Dimensionality of your model, can be '1D', '2D' or '3D'

    Returns:
        None
    """
    def _pred_tile(model_trained, tile):
        """Classify individual tiles."""
        if torch.cuda.is_available():
            pred = model_trained(tile.cuda()).cpu().detach().numpy()
        else:
            pred = model_trained(tile).detach().numpy()
        return pred

    if dim == '1D':
        out_shape = tiles_to_classify.shape[0]
    elif dim == '2D' or dim == '3D':
        out_shape = [tiles_to_classify.shape[idx] for idx in [0, 2, 3]]
    else:
        raise ValueError('Argument dim only supports values \
                         "1D", "2D" or "3D"')

    predicted_arr = np.empty(out_shape, dtype=np.int8)
    data_tensor = torch.from_numpy(tiles_to_classify)

    for tile in range(data_tensor.shape[0]):
        # This is a huge bottleneck for 1D rasters - for loop over each pixel
        if dim == '1D':
            pred = _pred_tile(model, data_tensor[tile, :, :][None, :])
            predicted_arr[tile] = (
                pred[0, :].argmax(0).squeeze().astype(np.uint8))
        elif dim == '2D' or dim == '3D':
            pred = _pred_tile(model, data_tensor[tile, :, :, :][None, :])
            predicted_arr[tile, :, :] = (
                pred[0, :, :, :].argmax(0).squeeze().astype(np.uint8))

    return predicted_arr


def write_rasterio(out_path: str, arr_classified: np.ndarray,
                   geoinfo: dict) -> None:
    """Write combined tiles into raster file using rasterio.

    Args:
        out_path: A path to the input file containing imagery.
        arr_classified: The array created by combining the classifed tiles.
        geoinfo: Dictionary with the original transform and crs.

    Returns:
        None
    """
    raise NotImplementedError
    print(f'Saving raster of size {arr_classified.shape} pixels.')
    with rasterio.open(out_path, 'w+',
                       width=arr_classified.shape[1],
                       height=arr_classified.shape[0],
                       count=1,
                       crs=geoinfo['crs'],
                       transform=geoinfo['transform']
                       # dtype=8BIT INT,
                       # 'COMPRESS=LZW'
                       ):
        # Write array to raster
        pass
    print(f'Raster saved succesfully to {out_path}')
    """
    driver = gdal.GetDriverByName('Gtiff')
    out_ds = driver.Create(out_path, xsize=arr.shape[1], ysize=arr.shape[0],
                           bands=1, eType=gdal.GDT_Byte,
                           options=['COMPRESS=LZW'])
    if geoinfo:
        out_ds.SetGeoTransform(geoinfo['geotransform'])
        out_ds.SetProjection(geoinfo['projection'])
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds = None
    print(f'Raster saved succesfully to {out_path}')
    """


if __name__ == '__main__':
    pass
