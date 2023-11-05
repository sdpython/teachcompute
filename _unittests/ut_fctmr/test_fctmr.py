import unittest
from teachcompute.ext_test_case import ExtTestCase
from teachcompute.fctmr import mapper, reducer, ffilter, take, combiner


class TestFctMr(ExtTestCase):
    def test_mapper(self):
        res = mapper(lambda x: x + 1, [4, 5])
        self.assertNotIsInstance(res, list)
        self.assertEqual(list(res), [5, 6])

    def test_ffilter(self):
        res = ffilter(lambda x: x % 2, [4, 5])
        self.assertNotIsInstance(res, list)
        self.assertEqual(list(res), [5])

    def test_take(self):
        res = take([4, 5, 6, 7, 8, 9], 2, 2)
        self.assertNotIsInstance(res, list)
        self.assertEqual(list(res), [6, 7])

    def test_reducer(self):
        res = reducer(lambda x: x[0], [("a", 1), ("b", 2), ("a", 3)], asiter=False)
        self.assertEqual(list(res), [("a", [("a", 1), ("a", 3)]), ("b", [("b", 2)])])
        res2 = reducer(
            lambda x: x[0], [("a", 1), ("b", 2), ("a", 3)], asiter=False, sort=False
        )
        self.assertEqual(
            list(res2), [("a", [("a", 1)]), ("b", [("b", 2)]), ("a", [("a", 3)])]
        )
        res3 = reducer(
            lambda x: x[0], [("a", 1), ("b", 2), ("a", 3)], asiter=True, sort=False
        )
        res4 = [(a, list(b)) for a, b in res3]
        self.assertEqual(
            list(res4), [("a", [("a", 1)]), ("b", [("b", 2)]), ("a", [("a", 3)])]
        )
        res5 = reducer(
            lambda x: x[0], [("a", 1), ("b", 2), ("a", 3)], asiter=True, sort=True
        )
        res6 = [(a, list(b)) for a, b in res5]
        self.assertEqual(list(res6), [("a", [("a", 1), ("a", 3)]), ("b", [("b", 2)])])

    def test_combiner(self):
        def c0(el):
            return el[0]

        ens1 = [("a", 1), ("b", 2), ("a", 3)]
        ens2 = [("a", 10), ("b", 20), ("a", 30)]
        res = combiner(c0, ens1, c0, ens2)
        exp = [
            (("a", 1), ("a", 10)),
            (("a", 1), ("a", 30)),
            (("a", 3), ("a", 10)),
            (("a", 3), ("a", 30)),
            (("b", 2), ("b", 20)),
        ]
        self.assertEqual(list(res), exp)

        ens1 = [("a", 1), ("b", 2)]
        ens2 = [("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="outer")
        exp = [(("a", 1), ("a", 10)), (("b", 2), None)]
        self.assertEqual(list(res), exp)

        ens1 = [("b", 2), ("a", 1)]
        ens2 = [("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="outer")
        exp = [(("a", 1), ("a", 10)), (("b", 2), None)]
        self.assertEqual(list(res), exp)

        ens1 = [("b", 2), ("a", 1)]
        ens2 = [("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="left")
        exp = [(("a", 1), ("a", 10)), (("b", 2), None)]
        self.assertEqual(list(res), exp)

        ens1 = [("b", 2), ("a", 1)]
        ens2 = [("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="right")
        exp = [(("a", 1), ("a", 10))]
        self.assertEqual(list(res), exp)

        ens1 = [("b", 2), ("a", 1)]
        ens2 = [("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="inner")
        exp = [(("a", 1), ("a", 10))]
        self.assertEqual(list(res), exp)

        ens1 = [("a", 1)]
        ens2 = [("b", 2), ("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="inner")
        exp = [(("a", 1), ("a", 10))]
        self.assertEqual(list(res), exp)

        ens1 = [("a", 1)]
        ens2 = [("b", 2), ("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="outer")
        exp = [(("a", 1), ("a", 10)), (None, ("b", 2))]
        self.assertEqual(list(res), exp)

        ens1 = [("a", 1)]
        ens2 = [("b", 2), ("a", 10)]
        res = combiner(c0, ens1, c0, ens2, how="right")
        exp = [(("a", 1), ("a", 10)), (None, ("b", 2))]
        self.assertEqual(list(res), exp)


if __name__ == "__main__":
    unittest.main()
