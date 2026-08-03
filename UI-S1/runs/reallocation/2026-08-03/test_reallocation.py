import unittest

import numpy as np

from reallocation_common import subset_bootstrap


class ReallocationContractTest(unittest.TestCase):
    def test_subset_bootstrap_positive_direction(self):
        rows=[{"id":f"r{i}","application":f"g{i%10}"} for i in range(100)]
        selected=[row["id"] for row in rows]
        left={row["id"]:True for row in rows}; right={row["id"]:False for row in rows}
        mapping={f"g{i}":i%5 for i in range(10)}
        result=subset_bootstrap(rows,selected,left,right,mapping)
        self.assertEqual(result["point_delta"],1.0)
        self.assertGreater(result["ci_99"][0],0)


if __name__=="__main__": unittest.main()
