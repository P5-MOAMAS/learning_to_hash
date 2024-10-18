import unittest

class TestExample(unittest.TestCase):
    def setUp(self):
        # Setting up the test
        self.someValue = 1
        return super().setUp()

    def tearDown(self):
        # Dispose of any objects that need to be disposed of. Is run independendent of test success or failure
        # Can be left undefined
        return super().tearDown()

    def test_me_pwease(self):
        self.assertTrue(1 == self.someValue)
