"""app/version.py -- single source of truth for the app's own identity:
product name, version, and author. Read by the window title, the About
dialog, and (via the PyInstaller spec) the packaged .app's own metadata --
one place to bump for a release, not three.
"""

PRODUCT_NAME = "ComplexDynamics"
VERSION = "0.5.0-beta"
AUTHOR = "Aiden Hill"
