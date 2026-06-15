# Two-Stage Autoignition and NTC Behaviour of DME/Air — MKWS project

Numerical study (Cantera, 0-D homogeneous reactor) of the low-temperature
autoignition of dimethyl ether (DME, CH3OCH3) in air: the Negative Temperature
Coefficient (NTC) region and two-stage (cool-flame) ignition.

Course: *Computational Methods in Combustion (MKWS)*, Warsaw University of Technology.

## Repository structure
```
dme_ntc_combustion.py   self-contained script: all simulations + 5 figures
report/                 LaTeX source (report.tex) and compiled report.pdf
figures/                generated figures (fig1..fig5)
data/                   raw results: sweep_*.csv (IDT vs T), history_*.csv
```

## Model
- Reactor: adiabatic, constant-pressure, 0-D homogeneous batch reactor.
- Mechanism: NUI Galway 2015 (`example_data/n-hexane-NUIG-2015.yaml`, ships with
  Cantera; 1268 species, 5336 reactions) — contains the full low-temperature DME
  sub-mechanism, so CH3OCH3 is used directly as the fuel.
- Conditions: T0 = 625–1300 K; p = 10/20/40 bar; phi = 0.5/1.0/2.0; air = O2:N2 = 1:3.76.
- IDT = time of max dT/dt; cool-flame delay = first dT/dt peak before it.

## Reproduce
```bash
pip install cantera numpy scipy pandas matplotlib
python dme_ntc_combustion.py          # writes fig1..fig5
cd report && pdflatex report.tex && pdflatex report.tex   # builds the PDF
```
Runtime ~5–10 min on 2 cores (the detailed mechanism is large).

## Key results
- Clear NTC region reproduced at all pressures.
- NTC weakens and shifts to higher T with pressure
  (valley 760→850 K; strength 1.48→1.26 from 10→40 bar).
- Richer mixtures ignite faster; phi has little effect on the low-T branch.
- Two-stage ignition: CH3OCH2O2 peroxy-radical pulse + CH2O build-up (cool flame)
  precede the main ignition.
