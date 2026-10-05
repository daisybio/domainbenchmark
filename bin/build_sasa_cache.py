#!/usr/bin/env python3
"""Entry point for the SASA_CACHE process. Lives in bin/ next to
extract_features.py, so `features` is importable the same way (the script's
directory is on sys.path) and sasa_cache.py's relative imports work."""
from features.sasa_cache import main

if __name__ == "__main__":
    main()