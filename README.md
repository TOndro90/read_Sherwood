# read_Sherwood
A very basic python code that can be used to read the spectra from the Sherwood simulation suite.

python sherwood_spectra_v3_save_all.py \
    ./spectra_40_2048/tauH1_lya_z3.0.dat \
    --save-all \
    --output-dir z3_los_txt

Small note from Sherwood simulation suite webpage:
If using Sherwood in a publication, please cite the Sherwood overview paper (Bolton et al. 2017, MNRAS, 464, 897) and consider adding the following acknowledgment:

The Sherwood simulations were performed with the Curie supercomputer, based at Le Très Grand Centre de Calcul (TGCC), and the DiRAC Data Analytic system at the University of Cambridge, operated by the University of Cambridge High Performance Computing Service on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The DiRAC Data Analytic system was funded by BIS National E-infrastructure capital grant ST/K001590/1, STFC capital grants ST/H008861/1 and ST/H00887X/1, and STFC DiRAC Operations grant ST/K00333X/1. DiRAC is part of the National E-Infrastructure.
