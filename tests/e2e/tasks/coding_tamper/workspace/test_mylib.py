import unittest
import mylib as m


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)


if __name__ == '__main__':
    unittest.main()
