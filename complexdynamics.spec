# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ComplexDynamics macOS .app bundle.

Build (from the repository root, after building cdx -- see CLAUDE.md):

    cmake -B cdx/build -S cdx -DCDX_BUILD_PYTHON=ON && cmake --build cdx/build
    pip install pyinstaller
    pyinstaller complexdynamics.spec

Produces dist/ComplexDynamics.app. This bundles the COMPILED cdx
extension (a Mach-O .so) for the platform PyInstaller runs on -- it
cannot cross-compile, and this bundle will only run on macOS. See
.github/workflows/build.yml for genuine per-platform builds (each
platform builds and packages on its own runner) and README.md for why
this file deliberately does not attempt a Windows build from here.
"""

import sys
from pathlib import Path

repo_root = Path(SPECPATH)
sys.path.insert(0, str(repo_root))
from app.version import AUTHOR, PRODUCT_NAME, VERSION   # noqa: E402
import PySide6   # noqa: E402

cdx_build_dir = repo_root / "cdx" / "build"
cdx_extension_candidates = sorted(cdx_build_dir.glob("cdx.cpython-*.so"))
if not cdx_extension_candidates:
    raise SystemExit(
        "cdx extension not found in cdx/build -- build it first:\n"
        "  cmake -B cdx/build -S cdx -DCDX_BUILD_PYTHON=ON && cmake --build cdx/build"
    )
cdx_extension = cdx_extension_candidates[0]

# This Anaconda-packaged PySide6 misreports its own Qt plugins directory:
# PyInstaller.utils.hooks.qt.pyside6_library_info.location['PluginsPath']
# resolves to /opt/anaconda3/plugins -- a completely different, Qt 5.15.2
# install shared conda-wide by unrelated GUI packages -- rather than this
# PySide6's own bundled Qt6 plugins under
# <PySide6 install dir>/Qt/plugins. PyInstaller's hook-PySide6.QtGui.py
# trusts that metadata and copies the WRONG (Qt5) .dylibs into the
# bundle's PySide6/Qt/plugins/platforms destination, silently shadowing
# what should have been the real Qt6 ones. Confirmed directly: the
# packaged app's Qt debug log reported every bundled platform plugin
# (including libqcocoa.dylib) as "uses incompatible Qt library (5.15.0)",
# even though QT_QPA_PLATFORM_PLUGIN_PATH correctly pointed AT that exact
# directory -- the path was right, the file's own contents were wrong.
pyside6_dir = Path(PySide6.__file__).parent
real_qt_plugins_dir = pyside6_dir / "Qt" / "plugins"
if not real_qt_plugins_dir.is_dir():
    raise SystemExit(f"expected PySide6's own Qt plugins at {real_qt_plugins_dir}")

a = Analysis(
    ["app/sandbox.py"],
    # cdx_build_dir MUST come before repo_root here. repo_root is needed
    # so app.* absolute imports resolve, but repo_root also contains the
    # cdx/ C++ SOURCE tree (include/, src/, python/, test/, build/, no
    # __init__.py) -- Python's import system treats that as a valid
    # (empty) implicit namespace package named "cdx". With repo_root
    # searched first, `import cdx` during Analysis resolved to that empty
    # namespace package instead of the real compiled extension, and
    # PyInstaller then froze in a placeholder "cdx" with none of its real
    # attributes -- confirmed as the cause of the packaged app's
    # "AttributeError: module 'cdx' has no attribute 'FamilyLibrary'"
    # crash: cdx/build was never on pathex at all, so the .so was only
    # ever bundled via the explicit `binaries` entry below, never actually
    # resolved as what `import cdx` means.
    pathex=[str(cdx_build_dir), str(repo_root)],
    binaries=[(str(cdx_extension), ".")],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

# Drop whatever hook-PySide6.QtGui.py collected under PySide6/Qt/plugins/
# (the wrong Qt5 binaries, per the note above) and re-add the SAME
# destination tree sourced from PySide6's own real Qt6 plugins directory
# instead. This is a substitution, not a trim -- an app with no platform
# plugin at all fails exactly as visibly as one with the wrong plugin, so
# skip-and-hope is not an option here.
_wrong_prefix = "PySide6/Qt/plugins/"
a.binaries = [b for b in a.binaries if not b[0].replace("\\", "/").startswith(_wrong_prefix)]
for _f in real_qt_plugins_dir.rglob("*.dylib"):
    _dest = f"PySide6/Qt/plugins/{_f.relative_to(real_qt_plugins_dir).as_posix()}"
    a.binaries.append((_dest, str(_f), "BINARY"))

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=PRODUCT_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=PRODUCT_NAME,
)

app = BUNDLE(
    coll,
    name=f"{PRODUCT_NAME}.app",
    icon=None,
    bundle_identifier="dev.complexdynamics.sandbox",
    info_plist={
        "CFBundleName": PRODUCT_NAME,
        "CFBundleDisplayName": PRODUCT_NAME,
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHumanReadableCopyright": f"© 2026 {AUTHOR}",
        "NSHighResolutionCapable": True,
        # UNSIGNED build (no Apple Developer ID here) -- Gatekeeper will
        # quarantine it on first download. README.md documents the
        # right-click -> Open workaround this requires; NOT something a
        # plist key can fix without an actual signing identity.
    },
)
