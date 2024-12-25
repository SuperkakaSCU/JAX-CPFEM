## Dual-phase steel
A crystal plasticity finite element demo of dual-phase steel considering two phases with different properties. This case can be directly cooperated with phase field simulation results.


## Execution
You can download `polycrystal_DPsteel/` folder and place it in the `applications/` folder of JAX-FEM, run
```bash
python -m applications.polycrystal_DPsteel.polycrystal_DPsteel_inhomo
```
from the `jax-fem/` diectory


## Results
Visualized with __ParaWiew__:
<p align="middle">
  <img src="docs/materials/polycrystal_DPsteel_phaseID.gif" width="360" />
  <img src="docs/materials/polycrystal_DPsteel_sigmaZZ.gif" width="360" />
</p>
<p align="middle">
    <em >Crystal plasticity: Dual-phase steel phase distribution (left) and z-z component of Cauchy stress (right).</em>
</p>
<br>