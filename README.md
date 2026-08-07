# ComplexDynamics

A rational-map dynamics sandbox, on the Riemann sphere throughout. Explore
Julia sets, parameter planes, basins of attraction, and Green's-function
potentials for a built-in family (Mandelbrot, multibrot, McMullen, Newton)
or a custom rational map you build term-by-term — poles and all.

Built to eventually support the simultaneous-approximation construction
(Fisher–Hill–Lazebnik–Thompson) this sandbox exists for in the first
place; see `ARCHITECTURE.md` for the layering and `CLAUDE.md` for the
non-obvious decisions behind the numerics.

## Download

Grab the build for your platform from the
[latest release](https://github.com/AidenHill03/CDS/releases/latest).
Each release ships three separately-built artifacts — `ComplexDynamics-macOS.zip`,
`ComplexDynamics-Windows.zip`, `ComplexDynamics-Linux.zip` — built and
launch-tested on their own platform in CI, not cross-compiled.

### macOS: unsigned build, right-click → Open

These builds are **not signed** with an Apple Developer ID. On first
launch, Gatekeeper will refuse to open `ComplexDynamics.app` with a
message like *"Apple could not verify..."* or *"is damaged and can't be
opened."* This is expected for an unsigned app, not a broken download.

To open it anyway:

1. Unzip `ComplexDynamics-macOS.zip`.
2. **Right-click** `ComplexDynamics.app` → **Open**.
3. Click **Open** again in the dialog that appears.

You only need to do this once — after the first right-click → Open, the
app launches normally (double-click) from then on.

### Windows / Linux

Unzip the archive and run `ComplexDynamics.exe` (Windows) or
`ComplexDynamics` (Linux) directly — both are onedir builds, so keep the
whole extracted folder together (the executable needs the files sitting
next to it).

## Building from source

Requires a C++17 compiler, CMake ≥ 3.15, and Python 3.12 with
`pybind11`/`PySide6`/`numpy` installed (see `requirements.txt`).

```bash
cd cdx && cmake -B build -DCDX_BUILD_PYTHON=ON && cmake --build build
./build/cdx_test                     # C++ tests
PYTHONPATH=build python python/demo.py
```

Interactive sandbox (from the repository root):

```bash
PYTHONPATH=cdx/build python -m app.sandbox
```

Packaging your own `.app`/`.exe`/binary (must be built ON the target
platform — PyInstaller cannot cross-compile):

```bash
pip install pyinstaller
PYTHONPATH=cdx/build pyinstaller complexdynamics.spec
```

## Development

`.github/workflows/build.yml` builds the C++ core, the pybind11
extension, runs both test suites, and packages the app on
macOS/Windows/Linux runners on every push and pull request. Every
packaged artifact must pass a launch self-test (a real window actually
comes up under that platform's real display plugin) before it's uploaded
— a bundle that builds but doesn't open is a failed CI run, not a warning.

Tagging a commit `vX.Y.Z` publishes that run's three artifacts as a
GitHub Release.
