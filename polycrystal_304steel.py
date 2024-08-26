## Hu: Benchmark for Differentiable crystal plasticity finite element 
## method accelerated by efficient GPU-computing 
## Hu: Case 3 for GPU Acceleration Study
## Domain: 0.016 mm x 0.016mm x 0.016mm with 8 grains
## Average Grain Size: 8 um x 8um x 8 um


## Case 3.1: 
### 1 -0.44841562434329973 0.16860994035907192 -0.3509983118612716 0.8045460216342235
### 2 -0.1721519777399991 0.1221784375558591 -0.7486299266954833 -0.628481788767607
### 3 0.1755902645854752 -0.8369269701982096 0.21157163334810386 0.4732428018470682
### 4 -0.8663740014241987 -0.187241901559973 0.4235298570897932 -0.18697331389780547
### 5 -0.9366201723048194 0.07470626885605275 0.34088334168529333 0.030986667887417784
### 6 -0.462754124567833 -0.3500710695060419 -0.3866988406362011 -0.7167795150120938
### 7 -0.6587757263479669 -0.16019499696083303 -0.3423669060109474 0.6504898208211397
### 8 0.1985059035197109 0.6384482594496135 0.05206231874156126 0.7418010118898696


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


from applications.GPU.models_304steel import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'polycrystal_304steel'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')
neper_folder = os.path.join(data_dir, f'neper/{case_name}')
ori_dir = os.path.join(csv_dir, f'grain_inds.txt')


def pre_processing(pf_args, neper_path='neper'):
    """We use Neper to generate polycrystal structure.
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
    pf_args['num_grains'] = 8
    pf_args['id'] = 0
    pf_args['domain_x'] = 0.016     ## domain size of cuboidal
    pf_args['domain_y'] = 0.016
    pf_args['domain_z'] = 0.016
    pf_args['num_oris'] = 8
    # pre_processing(pf_args, neper_path=f'neper/{case_name}')

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(neper_folder, f"domain0_mesh16.msh"))
    
    ## Hu: Represent which point(0~10648) belongs to which grains(0~99)
    cell_grain_inds = meshio_mesh.cell_data['gmsh:physical'][0] - 1
    grain_oris_inds = onp.arange(pf_args['num_oris'])
    #grain_oris_inds = onp.random.randint(pf_args['num_oris'], size=pf_args['num_grains'])
    #grain_oris_inds = onp.loadtxt(ori_dir)
    #grain_oris_inds = grain_oris_inds.astype(int)
    print("grain_oris_inds.shape",grain_oris_inds.shape)
    print(grain_oris_inds)


    ## Hu: Take elements from an array along an axis.
    ## set different points with specific ori. ## shape: (10648,)
    ## Hu: shape: (10648,) = 22*22*22 - cells
    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)
    print("cell_ori_inds",cell_ori_inds)

    ## Hu: def __init__(self, points, cells, ele_type='TET4')
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    print("No. of total mesh points:", mesh.points.shape)

    ## Hu: quat -- ('num_oris', 4) -- single crystal
    quat_file = os.path.join(csv_dir, f"quat.txt")
    # Hu: Load data from the file
    quat = onp.loadtxt(quat_file)[:pf_args['num_oris'], 1:]


    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    Lz = np.max(mesh.points[:, 2])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    # disps = np.linspace(0., 0.01*Lx, 51)
    # ts = np.linspace(0., 1., 51)

    # disps = np.linspace(0., 0.002*Lx, 11)
    # ts = np.linspace(0., 0.2, 11)
    ## Hu: medium strain 1.25%
    disps = np.linspace(0., 0.01*Lx, 51)
    ts = np.linspace(0., 0.1, 51)


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


    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn

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


    dirichlet_bc_info = [[corner, corner, bottom, top], 
                         [0, 1, 2, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]


    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, 
                                dirichlet_bc_info=dirichlet_bc_info, additional_info=(quat, cell_ori_inds))

    # print(problem.cells_list[0])
    # print(problem.cells_list[0].shape) -- (22*22*22, 8)
    #exit()

    results_to_save = []
    ## Hu: previous version: sol = np.zeros((problem.num_total_nodes, problem.vec))
    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
    #print(problem.fes[0].num_total_nodes) -- 12167
    #print(problem.fes[0].vec) -- 3
    
    ## Hu: previous version: params = problem.internal_vars['laplace']
    ## self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp]
    params = problem.internal_vars
    
    # exit()
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
        ## Hu: self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)
        ## dirichlet_bc_info : [location_fns, vecs, value_fns]
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
        
        ## Hu: update internal variables
        ## self.internal_vars = params
        problem.set_params(params)


        ## solver(problem, linear=False, precond=True, initial_guess=None, use_petsc=False, 
        ## petsc_options=None, lagrangian_solver=False, line_search_flag=False)
        sol_list = solver(problem, initial_guess=sol_list, use_petsc=False)   

        print(f"Computing stress...")
        sigma_cell_data = problem.compute_avg_stress(sol_list[0], params)[:, :, :]
        ## Macroscopic Cauchy Stress xx in the polycrystal
        print(f"Computing stress_xx...")
        sigma_cell_xx = sigma_cell_data[:, 0, 0]
        print(f"Computing stress_yy...")
        sigma_cell_yy = sigma_cell_data[:, 1, 1]
        print(f"Computing stress_zz...")
        sigma_cell_zz = sigma_cell_data[:, 2, 2]
        sigma_cell_xy = sigma_cell_data[:, 0, 1]
        sigma_cell_xz = sigma_cell_data[:, 0, 2]
        sigma_cell_yz = sigma_cell_data[:, 1, 2]

        print(f"Computing Von Mises Stress...")
        sigma_cell_von_mises_stress = ( 0.5*((sigma_cell_xx - sigma_cell_yy)**2.0 + (sigma_cell_yy - sigma_cell_zz)**2.0 + (sigma_cell_zz - sigma_cell_xx)**2.0) + \
                        + 3.0*(sigma_cell_xy**2.0 + sigma_cell_yz**2.0 + sigma_cell_xz**2.0))**0.5
        
        stress_xx_plot = np.append(stress_xx_plot, np.mean(sigma_cell_xx))
        stress_yy_plot = np.append(stress_yy_plot, np.mean(sigma_cell_yy))
        stress_zz_plot = np.append(stress_zz_plot, np.mean(sigma_cell_zz))
        von_mises_stress_plot = np.append(von_mises_stress_plot, np.mean(sigma_cell_von_mises_stress))
        print(f"stress_xx = {stress_xx_plot[-1]}")
        print(f"stress_yy = {stress_yy_plot[-1]}")
        print(f"stress_zz = {stress_zz_plot[-1]}")



        print(f"Updating int vars...")
        params = problem.update_int_vars_gp(sol_list[0], params)

        F_p_zz, slip_resistance_0, slip_0 = problem.inspect_interval_vars(params)
        print(f"stress = {sigma_cell_data[0]}, max stress = {np.max(sigma_cell_data)}")

        vtk_path = os.path.join(vtk_dir, f'u_{i:03d}.vtu')
        save_sol(problem.fes[0], sol_list[0], vtk_path, cell_infos=[('cell_ori_inds', cell_ori_inds), ('sigma_xx', sigma_cell_xx),('sigma_yy', sigma_cell_yy),('sigma_zz', sigma_cell_zz), ('von_Mises_stress', sigma_cell_von_mises_stress)])
    
    print("*************")
    print("grain_oris_inds:\n", grain_oris_inds)
    print("stress_xx:\n", onp.array(stress_xx_plot, order='F', dtype=onp.float64))
    print("stress_yy:\n", onp.array(stress_yy_plot, order='F', dtype=onp.float64))
    print("stress_zz:\n", onp.array(stress_zz_plot, order='F', dtype=onp.float64))
    print("von_mises_stress:\n", onp.array(von_mises_stress_plot, order='F', dtype=onp.float64))
    #print("verified stress_xx:\n",onp.array(stress_plot, order='F', dtype=onp.float64))
    print("*************")

if __name__ == "__main__":
    start_time_steel = time.time()
    problem()
    end_time_steel = time.time()
    run_time_steel = end_time_steel - start_time_steel
    print("Simulation time:", run_time_steel)
    print("This is for steel (FCC)")
