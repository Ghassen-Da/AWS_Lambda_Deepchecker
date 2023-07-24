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
        expected_data="""2023-07-24 19:53:54,753 - TheDeepChecker: deep_checker_result Logs - WARNING - Loss at cold start is considered poor and problematic: relative error of 32862193.857
2023-07-24 19:54:07,849 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer activation conv2d_7/relu are considered unstable with std of 0.25803529199566233 far from [0.5, 2.0]
2023-07-24 19:54:13,822 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer activation conv2d_7/relu are considered unstable with std of 0.28929210100820324 far from [0.5, 2.0]
2023-07-24 19:54:20,075 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer activation conv2d_7/relu are considered unstable with std of 0.29207487437564056 far from [0.5, 2.0]
2023-07-24 19:54:25,672 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.1519999504089355 < -4
2023-07-24 19:54:31,470 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.261000156402588 < -4
2023-07-24 19:54:37,551 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.057000160217285 < -4
2023-07-24 19:54:37,568 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.276000022888184 < -4
2023-07-24 19:54:37,650 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:54:43,518 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.122000217437744 < -4
2023-07-24 19:54:43,535 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.3379998207092285 < -4
2023-07-24 19:54:43,630 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:54:51,282 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.192999839782715 < -4
2023-07-24 19:54:51,282 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.056000232696533 < -4
2023-07-24 19:54:51,300 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.432000160217285 < -4
2023-07-24 19:54:51,382 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:54:56,854 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.252999782562256 < -4
2023-07-24 19:54:56,855 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.117000102996826 < -4
2023-07-24 19:54:56,871 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.4730000495910645 < -4
2023-07-24 19:54:56,952 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:02,659 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.301000118255615 < -4
2023-07-24 19:55:02,660 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.164000034332275 < -4
2023-07-24 19:55:02,677 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.49399995803833 < -4
2023-07-24 19:55:02,768 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:10,829 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.3480000495910645 < -4
2023-07-24 19:55:10,829 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.210999965667725 < -4
2023-07-24 19:55:10,846 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.521999835968018 < -4
2023-07-24 19:55:10,847 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.013999938964844 < -4
2023-07-24 19:55:10,930 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:17,098 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.394000053405762 < -4
2023-07-24 19:55:17,099 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.260000228881836 < -4
2023-07-24 19:55:17,116 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.556000232696533 < -4
2023-07-24 19:55:17,116 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.072000026702881 < -4
2023-07-24 19:55:17,199 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:23,645 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.432000160217285 < -4
2023-07-24 19:55:23,645 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.300000190734863 < -4
2023-07-24 19:55:23,662 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.579999923706055 < -4
2023-07-24 19:55:23,663 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.119999885559082 < -4
2023-07-24 19:55:23,750 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:31,057 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.465000152587891 < -4
2023-07-24 19:55:31,057 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.330999851226807 < -4
2023-07-24 19:55:31,075 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.5980000495910645 < -4
2023-07-24 19:55:31,075 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.159999847412109 < -4
2023-07-24 19:55:31,160 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:37,733 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.494999885559082 < -4
2023-07-24 19:55:37,734 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.363999843597412 < -4
2023-07-24 19:55:37,751 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.611000061035156 < -4
2023-07-24 19:55:37,752 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.195000171661377 < -4
2023-07-24 19:55:37,835 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:45,430 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.519000053405762 < -4
2023-07-24 19:55:45,431 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.39300012588501 < -4
2023-07-24 19:55:45,449 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.619999885559082 < -4
2023-07-24 19:55:45,451 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.2230000495910645 < -4
2023-07-24 19:55:45,537 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:53,134 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.539000034332275 < -4
2023-07-24 19:55:53,135 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.416999816894531 < -4
2023-07-24 19:55:53,152 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.625 < -4
2023-07-24 19:55:53,152 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.242000102996826 < -4
2023-07-24 19:55:53,244 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
2023-07-24 19:55:53,244 - TheDeepChecker: deep_checker_result Logs - WARNING - There is a lot of fluctuations, the smoothness of Loss is 0.474 > 0.5
2023-07-24 19:56:00,318 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.558000087738037 < -4
2023-07-24 19:56:00,318 - TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.436999797821045 < -4
2023-07-24 19:56:00,335 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.625 < -4
2023-07-24 19:56:00,335 - TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.252999782562256 < -4
2023-07-24 19:56:00,419 - TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing"""
        print('Test staring..')
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
        test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
        model = module.Model(x_train, y_train)
        data_under_test = interfaceData.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
        checker = DeepChecker(name='deep_checker_result', data=data_under_test, model=model, buffer_scale=10)
        checker.run_full_checks()
        log_file="deep_checker_result.log"
        with open(log_file, 'rb') as log_file:
            log_data = log_file.read().decode('utf-8')
        assert log_data==expected_data
        print('Test finished successfully.')
            

if __name__ == '__main__':
    unittest.main()