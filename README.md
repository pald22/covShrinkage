# covShrinkage
A Package for Shrinkage Estimation of Covariance Matrices

PURPOSE: To provide fast and accurate estimators of the covariance matrix based on linear and nonlinear shrinkage for general applications. 

LINEAR SHRINKAGE:
1) cov1Para: Linear shrinkage towards one-parameter matrix; all the variances are the same, all the covariances are zero. See Ledoit and Wolf (2004b).
2) cov2Para: Linear shrinkage towards two-parameter matrix; all the variances are the same, all the covariances are the same. See Ledoit (1995, Appendix B.1).
3) covCor: Linear shrinkage towards constant-correlation matrix; the target preserves the diagonal of the sample covariance matrix and all correlation coefficients are the same. See Ledoit and Wolf (2004a).
4) covDiag: Linear shrinkage towards diagonal matrix; the target preserves the diagonal of the sample covariance matrix and all the covariances are zero. See Ledoit (1995, Appendix B.2).
5) covMarket: Linear shrinkage towards a one-factor market model, where the factor is defined as the cross-sectional average of all the random variables; thanks to the idiosyncratic volatility of the residuals, the target preserves the diagonal of the sample covariance matrix. See Ledoit and Wolf (2003).

NONLINEAR SHRINKAGE:
6) GIS.m: Nonlinear shrinkage derived under the Symmetrized Kullback-Leibler loss, called geometric-inverse shrinkage (GIS). It can be viewed as geometrically averaging linear-inverse shrinkage (LIS) with quadratic-inverse shrinkage (QIS). See Ledoit and Wolf (2021, Remark 4.3).
7) LIS.m: Nonlinear shrinkage derived under Stein’s loss, called linear-inverse shrinkage (LIS). See Ledoit and Wolf (2021, Section 3).
8) QIS.m: Nonlinear shrinkage derived under Frobenius loss and its two cousins, Inverse Stein’s loss and Minimum Variance loss, called quadratic-inverse shrinkage (QIS). See Ledoit and Wolf (2021, Section 4.5). 

INPUT(S): Y (N*p): raw data matrix of N iid observations on p random variables.
Second optional input parameter: If the second (optional) parameter k is absent, not-a-number, or empty,then the algorithm demeans the data by default, and adjusts the effective sample size accordingly. If the user inputs k = 0, then no demeaning takes place; if user inputs k = 1, then it signifies that the data Y has already been demeaned. 

OUTPUT: sigmahat (p*p): invertible covariance matrix estimator.

REFERENCES:
a) Ledoit, O. (1995). Essays on Risk and Return in the Stock Market. PhD thesis, Massachusetts Institute of Technology, Sloan School of Management. Available online at http://dspace.mit.edu/handle/1721.1/11875. 
b) Ledoit, O. and Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. Journal of Empirical Finance, 10(5):603–621. doi:10.1016/S0927-5398(03)00007-0
c) Ledoit, O. and Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix. Journal of Portfolio Management, 30(4):110–119.
d) Ledoit, O. and Wolf, M. (2004b). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2):365–411. doi:10.1016/S0047-259X(03)00096-4
e) Ledoit, O. and Wolf, M. (2022). Quadratic shrinkage for large covariance matrices. Bernoulli. Forthcoming. Working paper version UZH ECON 335 available online at
https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html.

Copyright 2022:
- Olivier Ledoit (olivier.ledoit@econ.uzh.ch) for the Matlab version
- Michael Wolf  (michael.wolf@econ.uzh.ch) for the R version
- Patrick Ledoit (patrick.ledoit@yahoo.com) for the Python version
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
