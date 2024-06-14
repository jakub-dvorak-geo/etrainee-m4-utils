# test_preprocessing.py

import unittest
import sys
import os

# Add the directory containing the module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../src/etrainee_m4_utils'))
sys.path.append(module_path)
import preprocessing


class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define paths to test data
        cls.root_path = os.path.dirname(__file__)
        cls.img_path = os.path.join(cls.root_path,
                                    'testdata/test_img.tif')
        cls.ref_path = os.path.join(cls.root_path,
                                    'testdata/test_ref.tif')

        # Check that test data files exist
        if not os.path.exists(cls.img_path):
            raise IOError('Tests cannot run, missing testdata/test.img.tif.')
        if not os.path.exists(cls.ref_path):
            raise IOError('Tests cannot run, missing testdata/test_ref.tif.')

    # -----------------------------------------------------------------
    # Testing the read_rasterio method
    def test_rasterio_with_ref(self):
        # Load data
        loaded_data = preprocessing.read_rasterio(self.img_path, self.ref_path)
        # Checking dict keys
        self.assertIn('imagery', loaded_data)
        self.assertIn('reference', loaded_data)
        self.assertIn('crs', loaded_data)
        self.assertIn('transform', loaded_data)
        # Checking loaded shapes
        self.assertEqual(loaded_data['imagery'].shape, (1088, 1088, 54))
        self.assertEqual(loaded_data['reference'].shape, (1088, 1088, 1))

    def test_rasterio_without_ref(self):
        # Load data
        loaded_data = preprocessing.read_rasterio(self.img_path)
        # Checking dict keys
        self.assertIn('imagery', loaded_data)
        self.assertNotIn('reference', loaded_data)
        self.assertIn('crs', loaded_data)
        self.assertIn('transform', loaded_data)
        # Checking loaded shapes
        self.assertEqual(loaded_data['imagery'].shape, (1088, 1088, 54))

    def test_rasterio_wrong_path(self):
        nonexistent_path = os.path.join(self.root_path,
                                        'testdata/test_nonexistent.tif')
        # Check that the file really doesn't exist
        self.assertFalse(os.path.exists(nonexistent_path))
        # Check that loading the wrong file raises an IOError
        with self.assertRaises(IOError):
            preprocessing.read_rasterio(nonexistent_path)
        with self.assertRaises(IOError):
            preprocessing.read_rasterio(self.img_path, nonexistent_path)

    def test_rasterio_incompatible_crs(self):
        pass

    def test_rasterio_incompatible_transform(self):
        pass

    # -----------------------------------------------------------------
    # Testing the read_pavia_centre method

    # -----------------------------------------------------------------
    # Testing the split_into_tiles method
    def test_tile_working(self):
        # Process the imagery into tiles
        loaded_data = preprocessing.read_rasterio(self.img_path, self.ref_path)
        tiles = preprocessing.split_into_tiles(loaded_data, (64, 64), 32)

        # Check that arrays have the correct shape
        self.assertEqual(tiles['imagery'].shape, (1089, 64, 64, 54))
        self.assertEqual(tiles['reference'].shape, (1089, 64, 64, 1))

    def test_tile_rectangle(self):
        # Process the imagery into tiles
        loaded_data = preprocessing.read_rasterio(self.img_path, self.ref_path)
        tiles = preprocessing.split_into_tiles(loaded_data, (128, 64), 32)
        self.assertEqual(tiles['imagery'].shape, (363, 128, 64, 54))

    def test_tile_overlap_larger_than_shape(self):
        pass

    def test_tile_negative_shape_overlap(self):
        pass

    # -----------------------------------------------------------------
    # Testing the remove_nodata_tiles method
    def test_remove_nodata_success(self):
        pass

    def test_remove_nodata_reclass_tiles_zero_one(self):
        pass

    # -----------------------------------------------------------------
    # Testing the split_into_tiles method


"""
    def _test_remove_nodata(img_path, ref_path, t_shp, t_overlap):
        tiles = _test_split_into_tiles(img_path, ref_path, t_shp, t_overlap)
        filtered = remove_nodata_tiles(tiles, nodata_val=0)
        print('-------------------------------------------')
        print('Test remove nodata tiles')
        print(filtered.keys())
        print(filtered['imagery'].shape)
        if ref_path:
            print(filtered['reference'].shape)
        return filtered

    def _test_reclass_tile(img_path, ref_path, t_shp, t_overlap):
        filtered = _test_remove_nodata(img_path, ref_path, t_shp, t_overlap)
        reclassed = reclass_tiles_zero_one(filtered)
        print('-------------------------------------------')
        print('Test reclassify tiles')
        if ref_path:
            print(reclassed[0].keys())
            print(reclassed[0]['imagery'].shape)
            print(reclassed[0]['reference'].shape)
        else:
            print(reclassed.keys())
            print(reclassed['imagery'].shape)
        return reclassed
"""

if __name__ == '__main__':
    unittest.main()
