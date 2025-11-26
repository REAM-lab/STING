import unittest
import os
import numpy as np
from sting.main import run_ssm
from sting.utils.dynamical_systems import StateSpaceModel


class TestExampleSystems(unittest.TestCase):

    def testcase1(self):
        # Read the correct state-space matrices
        outputs_dir = os.path.join(os.getcwd(), "testcase1_ssm")
        true_ssm = StateSpaceModel.from_csv(outputs_dir)
        # Compute SSM with current version of STING
        input_dir = os.path.join(
            os.getcwd(), os.pardir, os.pardir, "examples", "testcase1"
        )
        sys, ssm = run_ssm(input_dir, write_outputs=False)
        # Compare SSMs to within numerical precision
        self.assertTrue(np.allclose(ssm.A, true_ssm.A))
        self.assertTrue(np.allclose(ssm.B, true_ssm.B))
        self.assertTrue(np.allclose(ssm.C, true_ssm.C))
        self.assertTrue(np.allclose(ssm.D, true_ssm.D))


if __name__ == "__main__":
    unittest.main()
