## Hu: Benchmark for Differentiable crystal plasticity finite element 
## method accelerated by efficient GPU-computing 
## Case 1: Test GPU acceleation version with MOOSE
## single crystal (BCC) case under compressive loadings
import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import meshio
import matplotlib.pyplot as plt
import time

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol


from applications.singlecrystal_tantalum.models_tantalum import CrystalPlasticity


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'singlecrystal_tantalum'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')
neper_folder = os.path.join(data_dir, f'neper/{case_name}')
ori_dir = os.path.join(csv_dir, f'grain_inds.txt')



def pre_processing(pf_args, neper_path='neper'):
    """
    We use Neper to generate polycrystal structure.
    Neper has two major functions: generate a polycrystal structure, and mesh it.
    See https://neper.info/ for more information.
    """
    neper_path = os.path.join(pf_args['data_dir'], neper_path)
    os.makedirs(neper_path, exist_ok=True)

    if not os.path.exists(os.path.join(neper_path, 'domain.msh')):
        print(f"You don't have neper mesh file ready, try generating them...")
        os.system(f'''neper -T -n {pf_args['num_grains']} -id {pf_args['id']} -regularization 0 -domain "cube({pf_args['domain_x']},\
                   {pf_args['domain_y']},{pf_args['domain_z']})" \
                    -o {neper_path}/domain -format tess,obj,ori''')
        os.system(f"neper -T -loadtess {neper_path}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
        os.system(f"neper -M -rcl 1 -elttype hex -faset faces {neper_path}/domain.tess")
    else:
        print(f"You already have neper mesh file.")




def problem():
    print(jax.lib.xla_bridge.get_backend().platform)


    pf_args = {}
    pf_args['data_dir'] = data_dir
    pf_args['num_grains'] = 1
    pf_args['id'] = 1
    pf_args['domain_x'] = 0.1  ## Unit: mm
    pf_args['domain_y'] = 0.1
    pf_args['domain_z'] = 0.1
    pf_args['num_oris'] = 1
    pre_processing(pf_args, neper_path=f'neper/{case_name}')


    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(neper_folder, f"mesh16.msh"))

    
    cell_grain_inds = meshio_mesh.cell_data['gmsh:physical'][0] - 1
    ## Hu: Assume a single crystal tantalum
    grain_oris_inds = onp.random.randint(pf_args['num_oris'], size=pf_args['num_grains'])
    print("grain_oris_inds", grain_oris_inds)
    ## Hu: Orientation index on each cell
    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
    ## Hu: Rotation applied on the single crystal
    quat = onp.array([[1, 0., 0., 0.]])


    ## Hu: Sizes of domain
    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    Lz = np.max(mesh.points[:, 2])


    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)


    ## Hu: Define applied loading
    disps = np.linspace(0., -0.0125*Lx, 51)
    ts = np.linspace(0., 12.5, 51)
    # disps = np.linspace(0., 0.01*Lx, 51)
    # ts = np.linspace(0., 1., 51)


    ## Hu: Define index of points and faces
    def corner(point):
        flag_x = np.isclose(point[0], 0., atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], Lz, atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def corner2(point):
        flag_x = np.isclose(point[0], 0., atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], 0., atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)


    def corner3(point):
        flag_x = np.isclose(point[0], Lx, atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], 0., atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)


    def corner4(point):
        flag_x = np.isclose(point[0], Lx, atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], Lz, atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def back(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)



    ## Hu: Define dirichlet B.C.
    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn


    dirichlet_bc_info = [[corner, corner, bottom, top], 
                         [0, 1, 2, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]

    '''
    dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
                          [0, 1, 2, 0, 1, 2], 
                          [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]
    '''



    ## Hu: Define CPFEM problem on top of JAX-FEM
    ## Xue, Tianju, et al. Computer Physics Communications 291 (2023): 108802.
    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, 
                                dirichlet_bc_info=dirichlet_bc_info, additional_info=(quat, cell_ori_inds))



    
    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
    params = problem.internal_vars
    
    
    ## Hu: Results to save
    #results_to_save = []
    stress_plot = np.array([])
    stress_xx_plot = np.array([])
    stress_yy_plot = np.array([])
    stress_zz_plot = np.array([])
    von_mises_stress_plot = np.array([])



    for i in range(len(ts) - 1):
        problem.dt = ts[i + 1] - ts[i]
        print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")

        dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
        ## Hu: Reset Dirichlet boundary conditions.
        ## Hu: Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        

        ## Hu: Set up internal variables of previous step for inner Newton's method
        ## self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp]
        problem.set_params(params)


        ## Hu: JAX-FEM's solver for outer Newton's method
        ## solver(problem, solver_options={})
        ## Examples:
        ## (1) solver_options = {'jax_solver': {}}
        ## (2) solver_options = {'umfpack_solver': {}}
        ## (3) solver_options = {'petsc_solver': {'ksp_type': 'bcgsl', 'pc_type': 'jacobi'}, 'initial_guess': some_guess}
        sol_list = solver(problem, solver_options={'jax_solver': {}, 'initial_guess': sol_list})


        ## Hu: Post-processing for aacroscopic Cauchy stress of each cell
        print(f"Computing stress...")
        sigma_cell_data = problem.compute_avg_stress(sol_list[0], params)[:, :, :]
        sigma_cell_xx = sigma_cell_data[:, 0, 0]
        sigma_cell_yy = sigma_cell_data[:, 1, 1]
        sigma_cell_zz = sigma_cell_data[:, 2, 2]
        sigma_cell_xy = sigma_cell_data[:, 0, 1]
        sigma_cell_xz = sigma_cell_data[:, 0, 2]
        sigma_cell_yz = sigma_cell_data[:, 1, 2]
        sigma_cell_von_mises_stress = ( 0.5*((sigma_cell_xx - sigma_cell_yy)**2.0 + (sigma_cell_yy - sigma_cell_zz)**2.0 + (sigma_cell_zz - sigma_cell_xx)**2.0) + \
                        + 3.0*(sigma_cell_xy**2.0 + sigma_cell_yz**2.0 + sigma_cell_xz**2.0))**0.5
        

        stress_xx_plot = np.append(stress_xx_plot, np.mean(sigma_cell_xx))
        stress_yy_plot = np.append(stress_yy_plot, np.mean(sigma_cell_yy))
        stress_zz_plot = np.append(stress_zz_plot, np.mean(sigma_cell_zz))
        von_mises_stress_plot = np.append(von_mises_stress_plot, np.mean(sigma_cell_von_mises_stress))
        print(f"Average Cauchy stress: stress_xx = {stress_xx_plot[-1]}, stress_yy = {stress_yy_plot[-1]}, stress_zz = {stress_zz_plot[-1]}, \
         vM_stress = {von_mises_stress_plot[-1]}, max stress = {np.max(sigma_cell_data)}")


        ## Hu: Update internal variables
        ## self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp]
        print(f"Updating int vars...")
        params = problem.update_int_vars_gp(sol_list[0], params)
        F_p_zz, slip_resistance_0, slip_0 = problem.inspect_interval_vars(params)


        ## Hu: Post-processing for visualization
        vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
        save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('cell_ori_inds', cell_ori_inds), ('sigma_xx', sigma_cell_xx),('sigma_yy', sigma_cell_yy),('sigma_zz', sigma_cell_zz), ('von_Mises_stress', sigma_cell_von_mises_stress)])
    

    print("*************")
    print("grain_oris_inds:\n", grain_oris_inds)
    print("stress_xx:\n", onp.array(stress_xx_plot, order='F', dtype=onp.float64))
    print("stress_yy:\n", onp.array(stress_yy_plot, order='F', dtype=onp.float64))
    print("stress_zz:\n", onp.array(stress_zz_plot, order='F', dtype=onp.float64))
    print("von_mises_stress:\n", onp.array(von_mises_stress_plot, order='F', dtype=onp.float64))
    print("*************")


if __name__ == "__main__":
    start_time_tantalum = time.time()
    problem()
    end_time_tantalum = time.time()
    run_time_tantalum = end_time_tantalum - start_time_tantalum
    print("Simulation time:", run_time_tantalum)
    print("This is for tantalum (BCC)")
