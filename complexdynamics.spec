# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ComplexDynamics desktop app -- macOS .app,
Windows onedir .exe, or Linux onedir binary, whichever platform this runs
on. PyInstaller cannot cross-compile, so each platform's artifact must be
built ON that platform -- see .github/workflows/build.yml, which runs
this exact spec on macos-latest/windows-latest/ubuntu-latest separately
to produce three genuine, native artifacts, not one file pretending to
work everywhere.

Build (from the repository root, after building cdx -- see CLAUDE.md):

    cmake -B cdx/build -S cdx -DCDX_BUILD_PYTHON=ON && cmake --build cdx/build --config Release
    pip install pyinstaller
    pyinstaller complexdynamics.spec

Produces dist/ComplexDynamics.app (macOS) or dist/ComplexDynamics/ (a
onedir folder containing ComplexDynamics.exe on Windows, or the
ComplexDynamics ELF binary on Linux, plus every DLL/.so it needs
alongside it).
"""

import sys
from pathlib import Path

repo_root = Path(SPECPATH)
sys.path.insert(0, str(repo_root))
from app.version import AUTHOR, PRODUCT_NAME, VERSION   # noqa: E402
import PySide6   # noqa: E402

cdx_build_dir = repo_root / "cdx" / "build"
# The compiled extension's suffix/location differs per platform: .so
# directly under build/ on macOS/Linux (single-config Makefile/Ninja
# generators), .pyd under a Release/ (or Debug/) subdirectory on Windows
# (Visual Studio's multi-config generator, which CMake's default Windows
# generator is) -- searching both shapes is what makes this spec work
# unmodified on all three CI runners, not just the platform it was
# originally written against.
cdx_extension_candidates = sorted(
    list(cdx_build_dir.glob("cdx.cpython-*.so")) +
    list(cdx_build_dir.glob("cdx.cp*.pyd")) +
    list(cdx_build_dir.glob("*/cdx.cpython-*.so")) +
    list(cdx_build_dir.glob("*/cdx.cp*.pyd"))
)
if not cdx_extension_candidates:
    raise SystemExit(
        "cdx extension not found in cdx/build -- build it first:\n"
        "  cmake -B cdx/build -S cdx -DCDX_BUILD_PYTHON=ON && cmake --build cdx/build --config Release"
    )
cdx_extension = cdx_extension_candidates[0]

a = Analysis(
    ["app/sandbox.py"],
    # cdx_extension.parent (NOT cdx_build_dir itself) MUST come before
    # repo_root here. repo_root is needed so app.* absolute imports
    # resolve, but repo_root also contains the cdx/ C++ SOURCE tree
    # (include/, src/, python/, test/, build/, no __init__.py) -- Python's
    # import system treats that as a valid (empty) implicit namespace
    # package named "cdx". With repo_root searched first (or with the
    # extension's real directory missing from the path at all), `import
    # cdx` during Analysis resolves to that empty namespace package
    # instead of the real compiled extension, and PyInstaller then
    # freezes in a placeholder "cdx" with none of its real attributes --
    # confirmed as the cause of an early packaged macOS build's
    # "AttributeError: module 'cdx' has no attribute 'FamilyLibrary'"
    # crash (cdx_build_dir was on pathex there, but the .so lived directly
    # under it, so that accidentally worked) and, separately, a v0.5.0-
    # beta.1 CI failure on Windows: cdx_build_dir alone does NOT contain
    # the extension there -- CMake's default Windows generator (Visual
    # Studio, multi-config) puts it under cdx/build/Release/ instead, so
    # cdx_build_dir being on pathex resolved nothing there either.
    # cdx_extension.parent is correct on every layout because it points
    # at wherever the file actually was FOUND (by the glob above), not
    # wherever it was assumed to be.
    pathex=[str(cdx_extension.parent), str(repo_root)],
    binaries=[(str(cdx_extension), ".")],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

# ---- conditional Qt-plugin fix: DETECT the mismatch, don't assume it ------
#
# On the local macOS dev machine this was written on, PyInstaller's own
# hook-PySide6.QtGui.py bundled the WRONG Qt platform plugins: this
# Anaconda-packaged PySide6 misreports its own plugins directory --
# PyInstaller.utils.hooks.qt.pyside6_library_info.location['PluginsPath']
# resolved to /opt/anaconda3/plugins, a completely unrelated, stray Qt
# 5.15.2 install shared conda-wide by OTHER GUI packages -- instead of
# this PySide6's own bundled Qt6 plugins under <PySide6 dir>/Qt/plugins.
# The hook trusted that metadata and copied the Qt5 .dylibs into the
# bundle, silently shadowing the real Qt6 ones (confirmed via
# QT_DEBUG_PLUGINS=1: every bundled platform plugin reported "uses
# incompatible Qt library (5.15.0)").
#
# THAT IS A LOCAL ANACONDA ENVIRONMENT DEFECT, not something a clean CI
# runner with pip-installed PySide6 has -- pip's PySide6 correctly
# reports its OWN plugins directory. Applying the substitution
# unconditionally would "fix" CI builds for a bug they never had (and,
# worse, silently mask a REAL future mismatch by always overwriting
# whatever PyInstaller collected, whether it needed it or not). So this
# detects the mismatch first: only substitute if PyInstaller's own
# reported PluginsPath is NOT actually inside this PySide6 install --
# i.e. only when there is something to fix.
pyside6_dir = Path(PySide6.__file__).parent
real_qt_plugins_dir = pyside6_dir / "Qt" / "plugins"

_reported_plugins_dir = None
try:
    from PyInstaller.utils.hooks.qt import pyside6_library_info
    _reported = pyside6_library_info.location.get("PluginsPath")
    if _reported:
        _reported_plugins_dir = Path(_reported).resolve()
except Exception as _exc:   # pragma: no cover -- defensive; see the print below
    print(f"complexdynamics.spec: could not query pyside6_library_info ({_exc}); "
          f"skipping the Qt-plugin mismatch check")

_mismatch = (
    _reported_plugins_dir is not None
    and real_qt_plugins_dir.is_dir()
    and not _reported_plugins_dir.is_relative_to(pyside6_dir.resolve())
)

if _mismatch:
    print(f"complexdynamics.spec: Qt plugin mismatch DETECTED -- PyInstaller collected "
          f"plugins from {_reported_plugins_dir}, outside this PySide6's own install at "
          f"{pyside6_dir}. Substituting the real ones.")
    _wrong_prefix = "PySide6/Qt/plugins/"
    a.binaries = [b for b in a.binaries if not b[0].replace("\\", "/").startswith(_wrong_prefix)]
    for _f in real_qt_plugins_dir.rglob("*"):
        if _f.is_file():
            _dest = f"PySide6/Qt/plugins/{_f.relative_to(real_qt_plugins_dir).as_posix()}"
            a.binaries.append((_dest, str(_f), "BINARY"))
else:
    print("complexdynamics.spec: Qt plugin directory looks consistent with this PySide6 "
          "install -- no substitution needed.")

# ---- Linux: explicitly bundle libxcb-cursor.so.0 -------------------------
#
# CI observed (v0.5.0-beta.1, ubuntu-latest): the packaged app aborted at
# launch with "qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is
# needed to load the Qt xcb platform plugin" / "Could not load the Qt
# platform plugin 'xcb' ... even though it was found" -- despite the
# system libxcb-cursor0 package being installed on the SAME runner just
# before packaging. PyInstaller's automatic Linux dependency walk (ldd-
# based, following direct link-time dependencies) does not pick this one
# up, because Qt's xcb platform plugin dlopen()s it itself at runtime
# rather than linking it directly -- the same class of "PyInstaller's
# automatic collection misses a plugin's own runtime dependency" issue
# the Qt-plugin substitution above exists for, just via dlopen instead of
# a wrong bundled version. Located via ldconfig on THIS SAME build
# machine, so the bundled copy's ABI matches every other bundled xcb-
# family library exactly, rather than trusting whatever happens to be (or
# not be) on an end user's system at run time.
if sys.platform.startswith("linux"):
    import subprocess
    try:
        _ldconfig_out = subprocess.run(
            ["ldconfig", "-p"], check=True, capture_output=True, text=True).stdout
    except Exception as _exc:
        print(f"complexdynamics.spec: ldconfig -p failed ({_exc}); cannot bundle "
              f"libxcb-cursor.so.0 explicitly")
        _ldconfig_out = ""
    _xcb_cursor_path = None
    for _line in _ldconfig_out.splitlines():
        if "libxcb-cursor.so.0" in _line and "=>" in _line:
            _xcb_cursor_path = _line.split("=>", 1)[1].strip()
            break
    if _xcb_cursor_path and Path(_xcb_cursor_path).is_file():
        print(f"complexdynamics.spec: bundling {_xcb_cursor_path} explicitly (Qt's "
              f"xcb plugin dlopen()s it; PyInstaller's automatic ldd-based walk does "
              f"not find it on its own).")
        a.binaries.append(("libxcb-cursor.so.0", _xcb_cursor_path, "BINARY"))
    else:
        print("complexdynamics.spec: libxcb-cursor.so.0 not found via ldconfig -- if "
              "the packaged app fails to load the xcb platform plugin, install "
              "libxcb-cursor0 on the build machine first.")

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

# BUNDLE (a .app with Info.plist) is a macOS-only PyInstaller construct.
# On Windows/Linux, `coll` above (a onedir folder: the .exe/binary plus
# every DLL/.so it needs alongside it) IS the final artifact -- there is
# no equivalent single-file bundle step to run.
if sys.platform == "darwin":
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
            # UNSIGNED build (no Apple Developer ID here) -- Gatekeeper
            # will quarantine it on first download. README.md documents
            # the right-click -> Open workaround this requires; NOT
            # something a plist key can fix without an actual signing
            # identity.
        },
    )
