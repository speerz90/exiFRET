# FRET Monte Carlo Simulator (ExiFRET-style)

Desktop application + library for calculating **FRET efficiency** between multiple donors and acceptors in arbitrary 2-D / 3-D geometries.

Based on the Monte Carlo method of:

- Corry, Jayatilaka & Rigby, *Biophys. J.* **89**:3822 (2005)
- Deplazes, Jayatilaka & Corry, *J. Biomed. Opt.* **17**:011005 (2012)

This is an independent, validated re-implementation (not the original FORTRAN / web ExiFRET program).

---

## Quick start (easiest way)

### 1. Install Python 3.8+

Any recent Python works. On Windows/macOS download from [python.org](https://www.python.org/downloads/).  
On Linux use your package manager.

### 2. Install the two required packages

```bash
pip install numpy matplotlib
```

(If you also want to run the validation suite against the original papers:)

```bash
pip install scipy
```

### 3. Make sure Tkinter is available (for the GUI)

| Platform              | Command                                      |
|-----------------------|----------------------------------------------|
| Ubuntu / Debian       | `sudo apt install python3-tk`                |
| Fedora                | `sudo dnf install python3-tkinter`           |
| macOS / Windows       | already included with official Python        |

### 4. Run the graphical interface

```bash
cd path/to/this/folder
python exifret_gui.py
```

A window opens with:

- **Left panel** – all simulation parameters (geometry, labelling, photophysics…)
- **Examples** menu – load published test cases with one click
- **▶ Run simulation** – computes mean FRET efficiency
- **Preview geometry** – shows one random configuration
- **Parameter sweep** tab – scan density, R₀, radius, etc.
- **File → Export** – save results or figures as CSV / PNG / PDF

---

## Command-line usage (no GUI needed)

```bash
# Simple run
python exifret.py --dimension 2 --nmer-size 1 --particles 300 --density 5e-4 --R0 50

# From a JSON settings file
python exifret.py --config my_settings.json

# User-supplied coordinates (one configuration)
python exifret.py --user-coords coords.txt --R0 50 --photons 20000
```

Coordinate file format (`coords.txt`):

```
# x   y   z   type
0.0  0.0  0.0  D
25.0 0.0  0.0  A
...
```

---

## Validation against the original papers

```bash
python validate.py              # run all tests
python validate.py pentamer     # only the isolated-pentamer analytic test
python validate.py solution plane labeling
```

Every test prints:

```
sim 0.XXX   ref 0.YYY   diff ±0.00Z   OK
```

Typical run time for the full suite is a few minutes.

---

## Project files

| File              | Purpose                                          |
|-------------------|--------------------------------------------------|
| `exifret.py`      | Core Monte Carlo engine (library + CLI)          |
| `exifret_gui.py`  | Desktop GUI (Tkinter + Matplotlib)               |
| `validate.py`     | Automated comparison with Corry / Deplazes results |
| `requirements.txt`| Python dependencies                              |

---

## Typical workflow inside the GUI

1. Choose an example from the **Examples** menu (or set parameters manually).
2. Click **Preview geometry** to inspect the arrangement.
3. Click **▶ Run simulation**.
4. Inspect the mean efficiency, the histogram over configurations, and (optionally) run a parameter sweep.
5. Export CSV or save the figure.

All parameters have tool-tips; hover over any label for a short explanation.

---

## Notes

- First run of a simulation may take a few extra seconds while Numba (if installed) compiles the fast kernel.
- Dense systems with “Avoid overlap” can take longer because of the annealing placement.
- For publication-quality statistics use ≥ 1000 configurations.
- The self-test in the Help menu (isolated pentamer vs analytic formula) is a quick sanity check.

Enjoy!
