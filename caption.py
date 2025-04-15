#!/usr/bin/env python3
"""
caption.py - CLI entry point.

    python caption.py --image path/to/image.jpg
    python caption.py --dir sample_images/

See `python caption.py --help` for all options.
"""

import sys

from captioner.cli import main

if __name__ == "__main__":
    sys.exit(main())
