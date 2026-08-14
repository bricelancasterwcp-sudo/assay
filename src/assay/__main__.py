"""Make ``python -m assay`` equivalent to the ``assay`` console script."""

import sys

from assay.cli import main

if __name__ == "__main__":
    sys.exit(main())
