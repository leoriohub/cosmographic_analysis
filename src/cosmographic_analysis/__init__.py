"""Cosmographic Analysis Package.

Testing Cosmological Isotropy through a Cosmographic approach
using Pantheon+ supernova data and hemispheric comparison.

IMPORTANT: The env vars below must be set before numpy/scipy import.
If the package is imported after numpy, the block in __init__.py
will have no effect — scripts must set these vars at the very top.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
