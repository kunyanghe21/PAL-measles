# Poisson Approximate Likelihood vs. Block Particle Filter  
### A Spatiotemporal Measles Model


> **Authors:** Kunyang He, Yize Hao & Edward L. Ionides  
> **Paper:** _Poisson Approximate Likelihood versus the Block Particle Filter for a Spatiotemporal Measles Model_ (working paper, 2025)

---

## Overview
This repository contains all code and Quarto notebooks (_\*.qmd_) required to reproduce the results in our paper Table 3.  

We provide two independent pipelines:

| Language | Folder | Toolkit | Purpose |
|----------|--------|---------|---------|
| **R**    | `pomp/`   | [`pomp`](https://kingaa.github.io/pomp/) & [`spatpomp`](https://github.com/spatPomp-org) | block particle filtering |
| **Python** | `wwr/` | [`wwr`](https://github.com/LorenzoRimella/PAL) | Poisson Approximate Likelihood (PAL) experiments |

The directory **`wwr/tidy_code_for_exp`** restructures all PAL experiments for clarity.

---