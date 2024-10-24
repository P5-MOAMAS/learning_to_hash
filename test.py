import unittest

from test_example import TestExample

def suite():
    suite = unittest.TestSuite()

    # Add tests
    suite.addTest(TestExample('test_me_pwease')) # Name the function you wish to be tested

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
