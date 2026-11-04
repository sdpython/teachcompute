import subprocess
import sys
import unittest
from pathlib import Path


class TestSetup(unittest.TestCase):
    def test_torch_paths_without_cuda_probe(self):
        setup = Path(__file__).resolve().parents[2] / "setup.py"
        script = """
import sys
from pathlib import Path
import setuptools
from distutils.core import run_setup
import torch


def unexpected_cuda_probe():
    raise AssertionError("Finding build paths must not probe the CUDA driver")


torch.cuda.is_available = unexpected_cuda_probe
distribution = run_setup(sys.argv[1], stop_after="config")
command = distribution.get_command_obj("build_ext")
command.use_cuda = False
command.ensure_finalized()
definitions = dict(
    argument[2:].split("=", 1)
    for argument in command.get_cmake_args("Release")
    if argument.startswith("-D")
)
includes = [Path(path) for path in definitions["TORCH_INCLUDE"].split(";")]
for header in ["torch/extension.h", "torch/torch.h"]:
    assert any((path / header).is_file() for path in includes), header
assert Path(definitions["TORCH_LIBRARIES_DIR"]).is_dir()
assert Path(definitions["CMAKE_PREFIX_PATH"]).is_dir()
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(setup)],
            cwd=setup.parent,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
