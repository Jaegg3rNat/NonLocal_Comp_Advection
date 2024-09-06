# NonLocal_Comp_Advection


This repository contains the codebase and some example data for the simulations, visualizations, and analyses conducted as part of the research presented in [Journal Name] under the title "Shear and transport in a flow environment determine spatial patterns and population dynamics in a model of nonlocal ecological competition."

## Contents
The repository is structured around:

Run_Flow_Integration.py: The main code that drives the simulation process. It implements the Pseudospectral Method of integration. It is responsible for run the time evolution for any given static velocity field $v_x,v_y$

Run_Snapshots.py: This script takes the output from the simulation and generates snapshots at various stages of the time evolution, allowing for a visual inspection of the simulation's progression..

Run_Animate.py: This script animates the generated snapshots, providing a dynamic view of the simulation's evolution over time.

Run_Segmentation.py: This script performs a segmentation analysis on the simulation results. In order to create the Example figure in the repository (and paper).
This python file runs the segmentation process implemented as follow:
   - View the Raw data of the patterns with a normalized to 20% o the maximum value.
   - Binarize the data to this treshold
   - Isolated a spot in interval region of the numerical space
   - Find the edges of the spot
   - Compute the the maximum distance between pairs of points
   - Find the orthogonal distance analitically
   - Search for the closest pair of points in opposite sides of the 'circle' and compute the distance between then.

Observations in this segmentation process:
   - When implementing for many data configurations as one single numerical loop. The last step could return points that are on the same side of the orthogonal line. This will create incorrect values for Taylor parameter. The solution at this moment, is either going by hand re-do this point increasing the parameter 'Dd' in the code or is already implemented filtering the wrong values from the final visualization (if they exists)
.

### Prerequisites
 - Python 3.x
 - NumPy, Matplotlib, sys, os, tqdm
 - fftpack from scipy (for fourier transform)
 - scikit-learn
 - seaborn
 - numba
 - h5py (for saving data)
 - cv2 (for animations)



 ### Instructions

 - The file Run_Flow_Integration needs two input parameters from sys.argv:
    the DamKholer variable mu. The Peclet intensity variable pe
 - The Run_Snapshots file will only work if Run_Flow_Integration integration was perfomed saving the density configurations over time
         The default of Run_Flow_Integration is to save only the final equilibrium configuration. To change this uncomment the section in the Run_Flow_Integration.py that saves the configurations over time.
   As an example we include the data for sinusoidal_mu800_pe_50 in the R_0.2 directory to be able to run the Run_Snapshots script.
   - Run_Animate works only after the snapshots figures were generated
   - 
