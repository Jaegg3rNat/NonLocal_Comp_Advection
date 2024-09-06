# NonLocal_Comp_Advection


This repository contains the codebase and some example data for the simulations, visualizations, and analyses conducted as part of the research presented in [Journal Name] under the title "Shear and transport in a flow environment determine spatial patterns and population dynamics in a model of nonlocal ecological competition."

## Contents
The repository is structured around:

Run_Integratio_Flow.py: The main code that drives the simulation process. It implements the Pseudospectral Method of integration. It is responsible for run the time evolution for any given static velocity field $v_x,v_y$

Snapshot Generation Script: This script takes the output from the simulation and generates snapshots at various stages, allowing for a visual inspection of the simulation's progression.

Animation Script: This script animates the generated snapshots, providing a dynamic view of the simulation's evolution over time.

Segmentation Analysis Script: This script performs a segmentation analysis on the simulation results, helping to extract and quantify key patterns and features.
