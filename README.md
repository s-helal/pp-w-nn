# Nonlinear Projection Pursuit with Neural Nets
## Files
Summary in chronological order
* `Gram-Schmidt.ipynb`: implementing and visualizing the Gram-Schmidt process
* `ML_DS Research Summer 2021.ipynb`: generating data of different dimensions and shapes
  - generating univariate normal data with inverse transform sampling, Box Muller transform, and Central Limit Theorem
  - generating multivariate normal data using univariate methods
  - generating $n$-dimensional blobs of data and comparing to `sklearn` methods
  - generating data uniform on $n$-dimensional spheres, cylinders, and cubes
* `data_generation.py`: data generation functions from `ML_DS Research Summer 2021.ipynb` (for importing to notebooks)
* `GUDHI practice.ipynb`: computing persistence diagrams and betti numbers
  - testing generating persistence diagrams with `gudhi`
  - computing Betti numbers
  - comparing our normal data generation functions to `numpy` methods
* `Autogradient.ipynb`: original linear projection pursuit (by Zhaoyang Shi)
  - using `tensorflow` and `tensorflow_manopt` to find optimal dimension reducing linear projections
* `Projection_Pursuit.ipynb`: modifying and testing `Autogradient.py`
* `old_NN.ipynb`: nonlinear projection pursuit with neural networks
  - building a `tensorflow` neural network to project 3D data to 2D 
  - implementing a `customAccuracy()` function in the network to incorporate the `RipsModel()` class from `Autogradient.ipynb`
  - tuning network by testing different learning rates and depths
  - trying to mitigate poor performance by scaling the data (whitening and log scaling)
  - testing on different datasets (1 cylinder, 2 cylinders and swiss roll)
* `NN.ipynb`: continuation of parameter tuning in `old_NN.ipynb`
  - testing various scaling options and modifying `RipsModel()` to fix poor performance on extreme-scale data
  - tuning parameters with grid search (learning rate, activation, number of layers, and weight initializers)
* `Shift_Inv_Dist.ipynb`: implementing scale (dilation) and shift-invariant bottleneck distance

## Sources
* [Interleaving/bottleneck distance](https://arxiv.org/pdf/2201.13012.pdf)
* [Shift-Invariant Bottleneck Distance](https://donsheehy.net/research/cavanna18computing.pdf)
* [Dilation invariant bottleneck distance](https://arxiv.org/pdf/2104.01672.pdf)
