"""Cosmographic Analysis Package.

Testing Cosmological Isotropy through a Cosmographic approach
using Pantheon+ supernova data and hemispheric comparison.
"""

import os
# Must be set before numpy/scipy imports to prevent OpenBLAS
# thread contention in multiprocessing workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
