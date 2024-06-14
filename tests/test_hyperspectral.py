# test_hyperspectral.py

import unittest
import sys
import os
import numpy as np

# Add the directory containing the module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../src/etrainee_m4_utils'))
sys.path.append(module_path)
import preprocessing
import hyperspectral


class TestHyperspectral(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        def _preprocess():
            # Load data
            loaded_data = preprocessing.read_rasterio(cls.img_path,
                                                      cls.ref_path)
            # flatten
            orig_shape = loaded_data['imagery'].shape
            flat_arrs = {
                'imagery': loaded_data['imagery'].reshape(
                    orig_shape[0]*orig_shape[1], orig_shape[2]),
                'reference': loaded_data['reference'].reshape(
                    orig_shape[0]*orig_shape[1])
            }
            # filter nodata
            filtered_arrs = {
                'imagery': flat_arrs['imagery'][flat_arrs['reference'] > 0],
                'reference': flat_arrs['reference'][flat_arrs['reference'] > 0]
            }
            # Subset the data
            x = filtered_arrs['imagery']
            y = filtered_arrs['reference']
            # extract spectral lib as mean band values
            class_vals = np.unique(y)
            num_classes = class_vals.shape[0]
            num_bands = x.shape[1]
            mean_bands = np.empty((num_classes, num_bands))
            for class_val in class_vals:
                mean_bands[class_val-1, :] = np.mean(
                    x[y == class_val], axis=0)
            return x, mean_bands

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

        # Preprocess imagery for testing
        cls.x, cls.y = _preprocess()
        print('input data shapes:')
        print(cls.x.shape)
        print(cls.y.shape)

    # -----------------------------------------------------------------
    # Testing the classify_SAM method
    def test_classify_SAM(self):
        classified = hyperspectral.classify_SAM(self.x, self.y, threshold=0.9)

        # Same number of pixels as in the imagery
        self.assertEqual(self.x.shape[0], classified.shape[0])
        # all classes are also in the spectral library
        self.assertLessEqual(np.unique(classified).shape[0], self.y.shape[0])

    # -----------------------------------------------------------------
    # Testing the classify_SID method
    def test_classify_SID(self):
        classified = hyperspectral.classify_SID(self.x, self.y, threshold=0.9)

        # Same number of pixels as in the imagery
        self.assertEqual(self.x.shape[0], classified.shape[0])
        # all classes are also in the spectral library
        self.assertLessEqual(np.unique(classified).shape[0], self.y.shape[0])


if __name__ == '__main__':
    unittest.main()
