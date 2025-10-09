# FYS-STK4155 - Project 1

Group members: Helene Lane and Manuela Leal Nader.

Description of the project:
     - The aim of the project is to implement and compare three regression methods, Ordinary Least Squares, Ridge and Lasso, applying them to Runge's function. We also compared different optimization and resampling methods.
     
Installation of packages required: Run the following command in your terminal. 
     - pip install -r requirements.txt

Code folder:
     - functions.py - functions used by the notebooks in this projects (runge, MSE, R2, OLS etc).
     - ordninary_least_squares.ipynb - exact OLS implementation. 
     - ridge_regression.ipynb - exact Ridge implementation.
     - gradient_descent.ipynb - implementation of plain gradient descent optimization.
     - updating_lr.ipynb - implementation of variations	of GD.
     - lasso_regression.ipynb - implementation of lasso regression with variations of GD.
     - stochastic_gd.ipynb - implementation of SCG.
     - figures_gd.ipynb - a notebook which creates the figures with the combined data from the gradient descent notebooks. 
     - bias_variance.ipynb - a notebook which does a bias-variance analysis using OLS and bootstrap for models with varying complexity.
     - kfold.ipynb - a notebook which implements k-fold cross validation on our OLS, Ridge and Lasso models. 