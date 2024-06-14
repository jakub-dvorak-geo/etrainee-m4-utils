# test_visualisation.py

import unittest
import sys
import os
import numpy as np

# Add the directory containing the module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../src/etrainee_m4_utils'))
sys.path.append(module_path)
import preprocessing
import visualisation


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

        cls.dataset = preprocessing.read_rasterio(cls.img_path, cls.ref_path)

    # -----------------------------------------------------------------
    # Testing the show_img_ref method
    def test_show_img_ref(self):
        # Load data
        visualisation.show_img_ref(self.dataset['imagery'][:, :, [25, 15, 5]],
                                   self.dataset['reference'],
                                   ds_name='bila_louka')
        # Checking dict keys
        # self.assertIn('imagery', loaded_data)

    # -----------------------------------------------------------------
    # Testing the confusion_matrix method
    def test_confusion_matrix(self):
        #visualisation.confusion_matrix()
        pass


if __name__ == '__main__':
    unittest.main()
