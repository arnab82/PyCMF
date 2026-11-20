# Singlet fission with multistate PDFT on TPSCI: prototype results

Pipeline: cMF clusters (one per chromophore) -> TPSCI states with state and transition RDMs (`tpsci_states.jl`) ->
L-PDFT from RDMs (`pycmf.lpdft`) -> diabatic LE / CT / TT analysis (`pycmf.diabatic`): direct LE -> TT coupling and
the CT-mediated (second-order superexchange) one. Energies in eV above the ground-state diabat, couplings in meV.

## Validation

L-PDFT on TPSCI singlets = pyscf-forge L-PDFT on CASCI singlets in the same orbitals, when TPSCI spans the full space:
ethylene dimer (4e, 8o), 6 singlets, tPBE / ftPBE / tBLYP / tPBE0: <= 3e-9 Eh; tetracene dimer (8e, 8o), 9 singlets,
tPBE / tPBE0: 2.6e-9 / 6.4e-9 Eh (needs `TPSCI_DELTA=4`: with 4 electrons per cluster, +-2 misses sectors of the CAS).
The CASCI reference needs conv_tol 1e-14 (PySCF's default leaves RDM errors ~1e-6).

## Ethylene dimer, 3.5 A, 6-31G, (4e, 8o)

Eclipsed: all LE -> TT couplings vanish by symmetry. Slip along C=C:

| slip | method | LE | CT | TT | direct | CT-mediated |
|---|---|---|---|---|---|---|
| 0.75 | TPSCI | 12.04 | 13.05 | 9.60 | 10.1 | 141 |
| 0.75 | L-PDFT tPBE | 7.61 | 6.77 | 10.22 | 11.8 | -50 |
| 0.75 | L-PDFT tPBE0 | 8.72 | 8.34 | 10.06 | 11.4 | -163 |
| 1.5 | TPSCI | 12.02 | 13.30 | 9.67 | 5.6 | 85 |
| 1.5 | L-PDFT tPBE | 7.61 | 6.75 | 10.29 | 10.6 | -29 |
| 1.5 | L-PDFT tPBE0 | 8.71 | 8.39 | 10.13 | 9.4 | -117 |

Size consistency at 15 A: E(1TT) = E(5TT) = 2 E(T1) in CASCI (9.629 eV) and MC-PDFT (tPBE 10.284, ftPBE 10.082,
tPBE0 10.120 eV): translated functionals place the local triplet ~0.3 eV above CASCI. A neutral open-shell local
singlet (CASCI 15.7 eV) gets the local-triplet energy (5.14 eV) in MC-PDFT: a state the on-top functional cannot tell
from the triplet; outside the 6-singlet model space here.

## Tetracene dimer, 3.5 A, 6-31G, frontier (8e, 8o) (HOMO-1..LUMO+1 of each monomer, no ooCMF)

Long-axis slip 1.2 A only: couplings vanish by symmetry. Long 1.2 + short 0.8 A:

| method | LE | CT | TT | direct | CT-mediated |
|---|---|---|---|---|---|
| TPSCI | 4.31 | 4.54 | 3.66 | 0.15 | -13.2 |
| L-PDFT tPBE | 1.67 | 1.41 | 2.75 | -0.10 | 2.9 |
| L-PDFT tPBE0 | 2.33 | 2.19 | 2.98 | -0.04 | 8.8 |

Experiment (tetracene): S1 ~2.3-2.4 eV, 2 T1 ~2.5 eV, CT above S1.

## Status

- In every method the CT-mediated coupling is 1-2 orders of magnitude larger than the direct one at this geometry,
  but its size and sign depend on where the method puts the CT states.
- L-PDFT with translated functionals in this minimal space: LE good with tPBE0 (2.33 eV), TT 0.2-0.5 eV too high,
  CT below LE (tPBE 0.26 eV, tPBE0 0.14 eV below). With CT below LE the superexchange formula does not apply (CT is a
  real intermediate), so the mechanism verdict hinges on the CT energetics.
- Next: SC-NEVPT2 reference on the same CASCI states (functional vs active-space error), larger frontier spaces
  (`--nfront 3, 4`), ooCMF orbitals, crystal geometry.
