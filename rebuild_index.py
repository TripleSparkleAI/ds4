#!/usr/bin/env python3
"""Shim: the index is built by track0_harness/card_tool.py index (CARD_STANDARD_v3.md). Usage unchanged: rebuild_index.py <stack-README-path> [--dry]"""
import os, subprocess, sys
sys.exit(subprocess.call([sys.executable, os.path.join(os.path.dirname(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()), "track0_harness", "card_tool.py"), "index", os.path.dirname(os.path.abspath(sys.argv[1])) if len(sys.argv) > 1 else "."] + [a for a in sys.argv[2:] if a == "--dry"]))
