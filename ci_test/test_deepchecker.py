import unittest
import os
os.chdir("ci_test")
import sys
sys.path.append('..')
import tensorflow as tf
from checkers import DeepChecker
import interfaceData as interfaceData
import data as data
from tensorflow.keras import datasets
import CNN_with_high_lr as module

class NeuraTest(unittest.TestCase):
    def test_neura(self):
        print('Test staring..')
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
        test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
        model = module.Model(x_train, y_train)
        data_under_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
        checker = DeepChecker(name='deep_checker_result', data=data_under_test, model=model, buffer_scale=10)
        checker.run_full_checks()
        print('Test finished.')
        log_file="deep_checker_result.log"
        with open(log_file, 'rb') as log_file:
            log_data = log_file.read().decode('utf-8')
            print(log_data)

if __name__ == '__main__':
    unittest.main()