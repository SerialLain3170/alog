"""
Runs Wan2.2's generate.py as if invoked directly, but imports cv2 first.

Real bug this works around: generate.py's own model loading (torch/torchvision,
loaded transitively while building WanAnimate) and cv2's bundled libpng/zlib end up
in the same process with conflicting native library versions — reproduced as
`libpng error: bad parameters to zlib` / cv2.imread returning None deep inside
wan/animate.py's prepare_source, well after the driving-video pose/face preprocessing
has already succeeded. Importing cv2 before torch gets a chance to load lets cv2's
bundled libpng/zlib win the process-wide symbol resolution, which avoids the crash.
This is a native-library load-order workaround, not a bug in our own pipeline code —
kept isolated here so it doesn't get lost inside wan_pipeline.py's subprocess-arg
plumbing.
"""

import os
import sys

import cv2  # noqa: F401

import runpy

if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    # runpy.run_path doesn't add the target script's directory to sys.path the way
    # `python generate.py` does — generate.py does `import wan` (its own local
    # package), which needs that directory on sys.path to resolve.
    sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
    runpy.run_path(sys.argv[0], run_name="__main__")
