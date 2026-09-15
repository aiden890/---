import unittest
import importlib.util
from pathlib import Path

class OutcomeTest(unittest.TestCase):
    def test_termination_reason_prioritizes_success_and_records_horizon(self):
        path=Path(__file__).with_name('outcome.py')
        self.assertTrue(path.exists(), 'outcome classifier not implemented')
        spec=importlib.util.spec_from_file_location('outcome',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.termination_reason(True,False,False,10,500),'success')
        self.assertEqual(module.termination_reason(False,True,False,10,500),'environment_done')
        self.assertEqual(module.termination_reason(False,False,True,10,500),'environment_truncated')
        self.assertEqual(module.termination_reason(False,False,False,500,500),'horizon')
        with self.assertRaises(ValueError):
            module.termination_reason(False,False,False,10,500)

if __name__=='__main__':
    unittest.main()
