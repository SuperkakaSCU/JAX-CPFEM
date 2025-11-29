## An efficient GPU-accelerated calibration of crystal plasticity model parameters by multi-objective gradient-based optimization with AD‐based sensitivities

On top of JAX-CPFEM package, the pipeline integrates AD-based sensitivity analysis with gradient-based optimization, offering a computationally efficient and accurate method for calibration.

__Case Study 1__: Single crystal copper (FCC) under tensile loading

<p align="middle">
  <img src="/docs/materials/cpfem_calibration_case1.png" width="800" />
</p>
<p align="middle">
    <em >Evolution of the objective function with the number of optimization iterations. And the comparison between reference and calibrated simulation results with different objective functions.</em>
</p>


__Case Study 5__: IN625 under tensile loading

<p align="middle">
  <img src="/docs/materials/cpfem_calibration_case5.tif" width="800" />
</p>
<p align="middle">
    <em >The calibration was performed on a RVE constructured based on EBSD measurement.</em>
</p>


## Reference
[1] https://doi.org/10.1115/1.4070536
