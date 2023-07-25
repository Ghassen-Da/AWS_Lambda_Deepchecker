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

class DeepCheckerTest(unittest.TestCase):
    
    def test_deepchecker(self):
        expected_data="""TheDeepChecker: deep_checker_result Logs - WARNING - Loss at cold start is considered poor and problematic: relative error of 32862193.857
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer activation conv2d_7/relu are considered unstable with std of 0.25803529199566233 far from [0.5, 2.0]
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer activation conv2d_7/relu are considered unstable with std of 0.28929210100820324 far from [0.5, 2.0]
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer activation conv2d_7/relu are considered unstable with std of 0.29207487437564056 far from [0.5, 2.0]
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.1519999504089355 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.261000156402588 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.057000160217285 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.276000022888184 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.122000217437744 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.3379998207092285 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.192999839782715 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.056000232696533 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.432000160217285 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.252999782562256 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.117000102996826 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.4730000495910645 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.301000118255615 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.164000034332275 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.49399995803833 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.3480000495910645 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.210999965667725 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.521999835968018 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.013999938964844 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.394000053405762 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.260000228881836 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.556000232696533 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.072000026702881 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.432000160217285 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.300000190734863 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.579999923706055 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.119999885559082 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.465000152587891 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.330999851226807 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.5980000495910645 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.159999847412109 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.494999885559082 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.363999843597412 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.611000061035156 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.195000171661377 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.519000053405762 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.39300012588501 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.619999885559082 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.2230000495910645 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.539000034332275 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.416999816894531 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.625 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.242000102996826 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing
TheDeepChecker: deep_checker_result Logs - WARNING - There is a lot of fluctuations, the smoothness of Loss is 0.474 > 0.5
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_6/kernel:0 change too slowly with magnitude update ratio of -4.558000087738037 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - Conv. layer weight conv2d_7/kernel:0 change too slowly with magnitude update ratio of -4.436999797821045 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_6/kernel:0 change too slowly with magnitude update ratio of -4.625 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - FC layer weight of dense_7/kernel:0 change too slowly with magnitude update ratio of -4.252999782562256 < -4
TheDeepChecker: deep_checker_result Logs - WARNING - The loss is no-or-slowly decreasing"""
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
        print('log data file:', log_data)
        print('expected data:', expected_data)
        differing_letters = []
        min_length = min(len(log_data), len(expected_data))
 
        for i in range(min_length):
            if log_data[i] != expected_data[i]:
                differing_letters.append(log_data[i])
                if len(differing_letters) == 1:
                    break

        print('hola')
        print(log_data==expected_data)
        print("".join(differing_letters))


        assert log_data==expected_data
        print('Test finished successfully.')




if __name__ == '__main__':
    unittest.main()