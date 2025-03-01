## Inverse Design of Grain Orientations via AD-based Sensitivities

This section presents a pipeline for inverse design, combining our end-to-end differentiable JAX-CPFEM with gradient-based optimization. We illustrate the power of this approach with an example of the inverse design of initial grain orientations in a polycrystalline metal, aiming to achieve targeted mechanical properties after a specific manufacturing process. 


### AD-based Sensitivity Analysis
To demonstrate the differentiable capabilities of JAX-CPFEM, we first introduced and verified a sensitivity analysis concerning the grain orientation of each grain in the CPFEM simulation, realized through automatic differentiation (AD). We consider a general copper (FCC) subjected to tensile loadings. The domain dimensions are 0.1 mm × 0.1 mm × 0.1 mm, discretized with 2 × 2 × 2 mesh. The sensitivity calculations for each mesh/cell (No. 1~8) using both AD and FDM-based methods are calculated for sanity check if the derivative is computed correctly.

The solver used in JAX-FEM is also attached to this file.

Place the downloaded file in the `applications/` folder of JAX-FEM, and then run
```bash
python -m applications.crystal_plasticity_OR_design.poly_sensitivity_rot_mesh2
```


### Inverse Design Case
The problem domain is a 0.1 mm × 0.1 mm × 0.1 mm polycrystal copper, discreated with 8 × 8 × 8 hexahedral mesh. The objective is to perform an inverse design of the initial crystal orientations of this 512 mesh, each representing different grains. The desired mechanical property is defined by a set of n points representing the local mechanical status (stress-strain curve), extracted from the corner cell adjacent to the origin (0, 0, 0).

The “optimizer” in the algorithm can use any off-the-shelf gradient-based optimization algorithms; for this problem, we used the limited-memory BFGS algorithm provided by the [SciPy](https://anaconda.org/anaconda/scipy) package as the optimizer. Please see this [paper](https://doi.org/10.1038/s41524-025-01528-2) for more details.


:fire: Here are the results shown in the paper:
<p align="middle">
  <img src="/docs/materials/fig09.jpg" width="800" />
</p>
<p align="middle">
    <em >Inverse design of the crystal orientation in polycrystalline copper. Subfigure (a) shows the targeted local mechanical properties (ground truth) of σ`zz` extract from the corner cell under different deformation stage x`i`, represented by black dots. The red dashed line indicates the JAX-CPFEM simulation outputs based on the initial guess for optimization, which significantly deviates from the targeted properties. The purple line represents JAX-CPFEM simulation results based on the crystal orientations designed by gradient-based optimization. The purple line closely aligns with the targeted properties, demonstrating the robustness of our pipeline. Subfigure (b) illustrates the percentage reduction in the objective function value falls within 0.4% with 32 steps.</em>
</p>


<p align="middle">
  <img src="/docs/materials/fig10.jpg" width="800" />
</p>
<p align="middle">
    <em >Inverse design of the initial crystal orientation of polycrystalline copper under deformations, involving three sequential rotations of Euler angles around the Z, Y, and X axes relative to their initial position, applied across all 512 mesh/grains using the differentiable JAX-CPFEM. </em>
</p>
