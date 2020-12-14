DFTSM
=====

Dynamic Functional Time Series Modelling in Python. This repo collect various python code and notebooks which runs the modelling for both simulation and an exapmle data set for image forecasting using functional decomposition techniques. 

This work is to examine the use of functional decomposition techniques such as FPCA, [1] and MAFR, [2] for forecasting time series of remote sensing imagery. We use the functional time series forecasting techniques of [3] but applies to bivariate functional data and compare the use of eigenfuncitions obtained through both FPCA and MAFR decompositions. 


Structure
*********
The structure of this repo is explained below:

* data - This directory is a place holder where data files can be kept. The files themselves are left out of the repo due to the size.
* notebooks - This directory hold the various jupyter notebooks for the data generation, modelling and figure generation for presentation of this work. 
* pres - This directory holds the beamer presentation files for the presentation of this work. 
* results - This directory holds the results of running of forecasting on both simulated and the example data sets. 
* src - This directory holds the python modules that we use within the notebooks. 

[1]: J. O. Ramsay and B. W. Silverman, Functional data analysis. New York (N.Y.): Springer Science+Business Media, 2010.

[2]: G. Hooker and S. Roberts, ‘Maximal autocorrelation functions in functional data analysis’, Stat Comput, vol. 26, no. 5, pp. 945–950, Sep. 2016, doi: 10.1007/s11222-015-9582-5.

[3]: H. Shang Lin, ‘ftsa: An R Package for Analyzing Functional Time Series’, The R Journal, vol. 5, no. 1, p. 64, 2013, doi: 10.32614/RJ-2013-006.

