#!/usr/bin/env python3
"""Back-compat shim: the demo runner now lives in hdc/cli.py and is
installed as the `hdc-demos` console script (see pyproject.toml).
Register new demos in hdc.cli.DEMOS, not here."""

from holo.cli import main

if __name__ == "__main__":
    main()
