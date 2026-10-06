"""Entry point for the PyInstaller executable (absolute imports only)."""

import sys

from e2eps.cli import main

if __name__ == "__main__":
    sys.exit(main())
