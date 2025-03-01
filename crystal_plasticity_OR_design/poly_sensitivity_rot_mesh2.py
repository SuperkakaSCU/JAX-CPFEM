## Hu: Sensitivity Analysis for CPFEM
## Hu: Sanity check of AD-based derivative with FDM-based derivative
## Hu: Parameters: [slip_resistance_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]

## Hu, Fanglei, et al. "Efficient GPU-computing simulation platform JAX-CPFEM for differentiable crystal plasticity finite element method." 
## npj Computational Materials 11.1 (2025): 46.

import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt
import time
import meshio


from jax_fem.solver import solver
from jax_fem.solver import implicit_vjp
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol
from jax_fem import logger

from jax.scipy.spatial.transform import Rotation as R

from applications.crystal_plasticity_OR_design.models_sens_rot import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

case_name = 'sensitivity'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')
neper_folder = os.path.join(data_dir, f'neper/{case_name}')



### Hu: Define a function transfering 3 euler rotation angles (ZYX) to R matrix for all cells
def rotate_theta_transfer(angle):
    alpha = angle[0]
    beta = angle[1]
    gamma = angle[2]
    euler_angles = [alpha, beta, gamma]

    ## Hu: Create a rotation object using the ZYX convention
    rotation = R.from_euler('zyx', euler_angles, degrees=True)

    ## Hu: Convert the euler angles' rotation to a rotation matrix
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix

rotate_theta_transfer_vmap = jax.vmap(rotate_theta_transfer)



### Hu: This is used for applying Rotation on Rotation matrix
def alpha_rotate_tensor_rank_2(R_a, R_b):
    ## Note: R_a is new applied rotation matrix
    ## Note: R_b is rot_mats applied on each cell
    return np.dot(R_a, R_b)

alpha_rotate_tensor_rank_2_vmap = jax.jit(jax.vmap(jax.vmap(alpha_rotate_tensor_rank_2, in_axes=(0, 0)), in_axes=(0, 0)))




def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        initial_guess = problem.initial_guess if hasattr(problem, 'initial_guess') else None
        # sol_list = solver(problem, {'jax_solver':{}, 'initial_guess': initial_guess, 'line_search_flag': True})
        # sol_list = solver(problem, solver_options={'petsc_solver':{}, 'initial_guess': initial_guess, 'line_search_flag': True}) 
        sol_list = solver(problem, solver_options={'petsc_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-5})   
        problem.set_initial_guess(sol_list)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred




def problem():
    print(jax.lib.xla_bridge.get_backend().platform)

    ## Hu: Definition of geometry 
    pf_args = {}
    pf_args['data_dir'] = data_dir
    pf_args['num_grains'] = 1
    pf_args['id'] = 1
    pf_args['domain_x'] = 0.1
    pf_args['domain_y'] = 0.1
    pf_args['domain_z'] = 0.1
    pf_args['num_oris'] = 1
    # pre_processing(pf_args, neper_path=f'neper/{case_name}')

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(neper_folder, f"mesh2.msh"))

    cell_grain_inds = meshio_mesh.cell_data['gmsh:physical'][0] - 1
    
    quat = onp.array([[1, 0., 0., 0.]])
    grain_oris_inds = onp.random.randint(pf_args['num_oris'], size=pf_args['num_grains'])
    print("grain_oris_inds", grain_oris_inds)

    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)
    print("cell_ori_inds.shape", cell_ori_inds.shape)
    print(cell_ori_inds)
    
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    Lz = np.max(mesh.points[:, 2])


    ## Hu: Definition of B.C. 
    ## 2% strain
    disps = np.linspace(0., 0.02*Lx, 11)
    ts = np.linspace(0., 2.0, 11)


    def corner(point):
        flag_x = np.isclose(point[0], 0., atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], 0., atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], Lz, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn

    dirichlet_bc_info = [[corner, corner, bottom, top], 
                         [0, 1, 2, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]
    
    cell_ori_inds = onp.zeros(len(mesh.cells), dtype=onp.int32)


    ## Hu: Definition of JAX-FEM problem
    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                                additional_info=(quat, cell_ori_inds))
    
    
    ## Hu: AD wrapper
    fwd_pred = ad_wrapper(problem)



    def simulation(alpha):
        print("**************")
        ### Hu: alpha is rotation matrix applied on each cell
        print("alpha=",alpha)

        ## Hu: Transfer euler angle on each cell to rotation matrix on each cell
        # (num_cells, 3, 3)
        alpha_scale = rotate_theta_transfer_vmap(alpha)
        #print("alpha_scale", alpha_scale.shape)


        ### Hu: Map rotation matrix for each cell on each quad point
        ### Hu: (num_cells, num_quads, 3, 3)
        alpha_scale_gp = np.repeat(alpha_scale[:, None, :], problem.fes[0].num_quads, axis=1)
        #print("alpha_scale_gp.shape", alpha_scale_gp.shape)
        
        
        ### Hu: internal parameters defined on each quad point
        ## self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
        params = problem.internal_vars
        

        ### Hu: params[3] is rot_mats_gp, initial rotation matrix defined on each quad point
        ## Hu: (num_cells, num_quads, dim, dim)
        params[3] = alpha_rotate_tensor_rank_2_vmap(alpha_scale_gp, params[3])
        print("params[3].shape", params[3].shape)
        
        
        ### Hu: objective function
        obj_func = 0.
        stress_plot = []
        
        
        for i in range(10):
            problem.dt = ts[i + 1] - ts[i]
            print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")

            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
            problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)

            sol_list = fwd_pred(params)
            sol = sol_list[0]

            print(f"Computing stress...")
            sigma_cell_data = problem.compute_avg_stress(sol_list[0], params)[:, :, :]
            print(f"Computing stress_zz...")
            sigma_cell_zz = sigma_cell_data[:, 2, 2]

            ## Hu: The corner mesh/cell labeled ‘1’, located at the bottom-left corner 
            ## in the x-y plane of the domain, adjacent to the origin (0, 0, 0)
            stress_zz = sigma_cell_data[0, 2, 2]

            params = problem.update_int_vars_gp(sol, params)

            stress_plot.append(stress_zz)
            print(f"\nStep {i + 1} in {len(ts) - 1}, stress_zz = {stress_zz}")

            
        
        print("!!!!Hey here!!!!")
        return stress_zz


    ## Hu: Transfer jax.numpy to numpy -- objective function
    def objective_wrapper(x):
        print("***Calling objective_wrapper***")
        print(f"x = {x}")
        x = np.array(x)

        sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
        problem.set_initial_guess(sol_list)

        ## Hu: initialize the global variables stored by JAX between calling forward CPFEM
        problem.custom_init(quat, cell_ori_inds)

        obj_val = simulation(x)
        print(f"Finishes objective, obj_val = {obj_val}")

        obj_val = onp.array(obj_val)

        print("***Finishing objective***")
        return onp.array(obj_val, order='F', dtype=onp.float64)



    ## Hu: Define the derivative function
    print("***Define the derivative***")
    grads_func = jax.grad(simulation)


    ## Hu: Transfer jax.numpy to numpy -- derivative function
    def derivative_wrapper(x):
        print("***Calling derivative_wrapper***")
        x = np.array(x)

        sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
        problem.set_initial_guess(sol_list)

        problem.custom_init(quat, cell_ori_inds)

        grads = grads_func(x)

        print("***Finishing derivative***")
        # 'L-BFGS-B' & 'BFGS' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(grads, order='F', dtype=onp.float64)



    
    ############################## Sanity Check:
    ## pt: 'z-y-x' euler rotation on each cell - (num_cells, 3)
    #detla_alpha = onp.pi/1000
    pt = onp.array([[30., 40., 50.]])
    print("pt.shape",pt.shape)
    pt_gp = onp.repeat(pt, 8, axis=0)
    print("pt_gp.shape",pt_gp.shape)
    
    ### Hu: AD-based sensitivity
    start_time_AD = time.time()
    print("AD")
    r0 = derivative_wrapper(pt_gp)
    end_time_AD = time.time()
    print("JAX AD", r0)
    print("Running 1 AD:", end_time_AD - start_time_AD)



    ### Hu: FDM-based sensitivity
    ## While the primary benefit of AD in handling complex non-linear models was often highlighted, 
    ## the computational cost comparison between AD and non-AD approaches is equally important, yet 
    ## frequently overlooked. In this CPFEM problem, the number of crystal orientations in each cell 
    ## typically exceeds that of the objective function, necessitating running forward CPFEM models 
    ## twice as many times as the number of design parameters.
    detla_alpha = 0.1
    print("1st FDM")
    start_time = time.time()
    pt_upper1 = pt_gp + onp.array([[detla_alpha, 0., 0.], 
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper1)
    tt_upper1 = objective_wrapper(pt_upper1)
    end_time = time.time()

    pt_below1 = pt_gp - onp.array([[detla_alpha, 0., 0.], 
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below1 = objective_wrapper(pt_below1)


    print("2nd FDM")
    pt_upper2 = pt_gp + onp.array([[0., detla_alpha, 0.], 
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper2)
    tt_upper2 = objective_wrapper(pt_upper2)

    pt_below2 = pt_gp - onp.array([[0., detla_alpha, 0.], 
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below2 = objective_wrapper(pt_below2)


    print("3rd FDM")
    pt_upper3 = pt_gp + onp.array([[ 0., 0., detla_alpha,], 
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper3)
    tt_upper3 = objective_wrapper(pt_upper3)

    pt_below3 = pt_gp - onp.array([[ 0., 0., detla_alpha,], 
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below3 = objective_wrapper(pt_below3)
    
    print("1st cell")
    print("1st FDM", (tt_upper1 - tt_below1)/(2*detla_alpha))
    print("2nd FDM", (tt_upper2 - tt_below2)/(2*detla_alpha))
    print("3rd FDM", (tt_upper3 - tt_below3)/(2*detla_alpha))

    print("4th FDM")
    pt_upper4 = pt_gp + onp.array([[0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper4)
    tt_upper4 = objective_wrapper(pt_upper4)

    pt_below4 = pt_gp - onp.array([[0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below4 = objective_wrapper(pt_below4)


    print("5th FDM")
    pt_upper5 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper5)
    tt_upper5 = objective_wrapper(pt_upper5)

    pt_below5 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below5 = objective_wrapper(pt_below5)



    print("6th FDM")
    pt_upper6 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper6)
    tt_upper6 = objective_wrapper(pt_upper6)

    pt_below6 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below6 = objective_wrapper(pt_below6)

    print("2nd cell")
    print("4 FDM", (tt_upper4 - tt_below4)/(2*detla_alpha))
    print("5 FDM", (tt_upper5 - tt_below5)/(2*detla_alpha))
    print("6 FDM", (tt_upper6 - tt_below6)/(2*detla_alpha))



    print("7th FDM")
    pt_upper7 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper7)
    tt_upper7 = objective_wrapper(pt_upper7)

    pt_below7 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below7 = objective_wrapper(pt_below7)


    print("8th FDM")
    pt_upper8 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper8)
    tt_upper8 = objective_wrapper(pt_upper8)

    pt_below8 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below8 = objective_wrapper(pt_below8)



    print("9th FDM")
    pt_upper9 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper9)
    tt_upper9 = objective_wrapper(pt_upper9)

    pt_below9 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below9 = objective_wrapper(pt_below9)

    print("3rd cell")
    print("7 FDM", (tt_upper7 - tt_below7)/(2*detla_alpha))
    print("8 FDM", (tt_upper8 - tt_below8)/(2*detla_alpha))
    print("9 FDM", (tt_upper9 - tt_below9)/(2*detla_alpha))




    print("10th FDM")
    pt_upper10 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper10)
    tt_upper10 = objective_wrapper(pt_upper10)

    pt_below10 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below10 = objective_wrapper(pt_below10)


    print("11th FDM")
    pt_upper11 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper11)
    tt_upper11 = objective_wrapper(pt_upper11)

    pt_below11 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below11 = objective_wrapper(pt_below11)



    print("12th FDM")
    pt_upper12 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper12)
    tt_upper12 = objective_wrapper(pt_upper12)

    pt_below12 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below12 = objective_wrapper(pt_below12)

    print("4th cell")
    print("10 FDM", (tt_upper10 - tt_below10)/(2*detla_alpha))
    print("11 FDM", (tt_upper11 - tt_below11)/(2*detla_alpha))
    print("12 FDM", (tt_upper12 - tt_below12)/(2*detla_alpha))


    print("13th FDM")
    pt_upper13 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper13)
    tt_upper13 = objective_wrapper(pt_upper13)

    pt_below13 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below13 = objective_wrapper(pt_below13)


    print("14th FDM")
    pt_upper14 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper14)
    tt_upper14 = objective_wrapper(pt_upper14)

    pt_below14 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below14 = objective_wrapper(pt_below14)



    print("15th FDM")
    pt_upper15 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper15)
    tt_upper15 = objective_wrapper(pt_upper15)

    pt_below15 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below15 = objective_wrapper(pt_below15)


    print("5th cell")
    print("13 FDM", (tt_upper13 - tt_below13)/(2*detla_alpha))
    print("14 FDM", (tt_upper14 - tt_below14)/(2*detla_alpha))
    print("15 FDM", (tt_upper15 - tt_below15)/(2*detla_alpha))


    print("16th FDM")
    pt_upper16 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper16)
    tt_upper16 = objective_wrapper(pt_upper16)

    pt_below16 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below16 = objective_wrapper(pt_below16)


    print("17th FDM")
    pt_upper17 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper17)
    tt_upper17 = objective_wrapper(pt_upper17)

    pt_below17 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below17 = objective_wrapper(pt_below17)



    print("18th FDM")
    pt_upper18 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper18)
    tt_upper18 = objective_wrapper(pt_upper18)

    pt_below18 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])
    tt_below18 = objective_wrapper(pt_below18)

    print("6th cell")
    print("16 FDM", (tt_upper16 - tt_below16)/(2*detla_alpha))
    print("17 FDM", (tt_upper17 - tt_below17)/(2*detla_alpha))
    print("18 FDM", (tt_upper18 - tt_below18)/(2*detla_alpha))



    print("19th FDM")
    pt_upper19 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.]])
    print(pt_upper19)
    tt_upper19 = objective_wrapper(pt_upper19)

    pt_below19 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.],
                                   [0., 0., 0.]])
    tt_below19 = objective_wrapper(pt_below19)


    print("20th FDM")
    pt_upper20 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.]])
    print(pt_upper20)
    tt_upper20 = objective_wrapper(pt_upper20)

    pt_below20 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., detla_alpha, 0.],
                                   [0., 0., 0.]])
    tt_below20 = objective_wrapper(pt_below20)



    print("21th FDM")
    pt_upper21 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.]])
    print(pt_upper21)
    tt_upper21 = objective_wrapper(pt_upper21)

    pt_below21 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., detla_alpha],
                                   [0., 0., 0.]])
    tt_below21 = objective_wrapper(pt_below21)


    print("7th cell")
    print("19 FDM", (tt_upper19 - tt_below19)/(2*detla_alpha))
    print("20 FDM", (tt_upper20 - tt_below20)/(2*detla_alpha))
    print("21 FDM", (tt_upper21 - tt_below21)/(2*detla_alpha))


    print("22th FDM")
    pt_upper22 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.]])
    print(pt_upper22)
    tt_upper22 = objective_wrapper(pt_upper22)

    pt_below22= pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [detla_alpha, 0., 0.]])
    tt_below22 = objective_wrapper(pt_below22)


    print("23th FDM")
    pt_upper23 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [ 0., detla_alpha, 0.]])
    print(pt_upper23)
    tt_upper23 = objective_wrapper(pt_upper23)

    pt_below23 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [ 0., detla_alpha, 0.]])
    tt_below23 = objective_wrapper(pt_below23)



    print("24th FDM")
    pt_upper24 = pt_gp + onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [ 0., 0., detla_alpha]])
    print(pt_upper24)
    tt_upper24 = objective_wrapper(pt_upper24)

    pt_below24 = pt_gp - onp.array([[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [ 0., 0., detla_alpha]])
    tt_below24 = objective_wrapper(pt_below24)


    print("8th cell")
    print("22 FDM", (tt_upper22 - tt_below22)/(2*detla_alpha))
    print("23 FDM", (tt_upper23 - tt_below23)/(2*detla_alpha))
    print("24 FDM", (tt_upper24 - tt_below24)/(2*detla_alpha))
    

    print("************* Sanity Check of AD-based sensitivity")
    print("Running 1 forward:", end_time - start_time)
    print("zyx rotation:", pt)  
    print("1st cell")
    print("1st FDM", (tt_upper1 - tt_below1)/(2*detla_alpha))
    print("2nd FDM", (tt_upper2 - tt_below2)/(2*detla_alpha))
    print("3rd FDM", (tt_upper3 - tt_below3)/(2*detla_alpha))

    print("2nd cell")
    print("4th FDM", (tt_upper4 - tt_below4)/(2*detla_alpha))
    print("5th FDM", (tt_upper5 - tt_below5)/(2*detla_alpha))
    print("6th FDM", (tt_upper6 - tt_below6)/(2*detla_alpha))

    print("3rd cell")
    print("7 FDM", (tt_upper7 - tt_below7)/(2*detla_alpha))
    print("8 FDM", (tt_upper8 - tt_below8)/(2*detla_alpha))
    print("9 FDM", (tt_upper9 - tt_below9)/(2*detla_alpha))

    print("4th cell")
    print("10 FDM", (tt_upper10 - tt_below10)/(2*detla_alpha))
    print("11 FDM", (tt_upper11 - tt_below11)/(2*detla_alpha))
    print("12 FDM", (tt_upper12 - tt_below12)/(2*detla_alpha))

    print("5th cell")
    print("13 FDM", (tt_upper13 - tt_below13)/(2*detla_alpha))
    print("14 FDM", (tt_upper14 - tt_below14)/(2*detla_alpha))
    print("15 FDM", (tt_upper15 - tt_below15)/(2*detla_alpha))

    print("6th cell")
    print("16 FDM", (tt_upper16 - tt_below16)/(2*detla_alpha))
    print("17 FDM", (tt_upper17 - tt_below17)/(2*detla_alpha))
    print("18 FDM", (tt_upper18 - tt_below18)/(2*detla_alpha))

    print("7th cell")
    print("19 FDM", (tt_upper19 - tt_below19)/(2*detla_alpha))
    print("20 FDM", (tt_upper20 - tt_below20)/(2*detla_alpha))
    print("21 FDM", (tt_upper21 - tt_below21)/(2*detla_alpha))

    print("8th cell")
    print("22 FDM", (tt_upper22 - tt_below22)/(2*detla_alpha))
    print("23 FDM", (tt_upper23 - tt_below23)/(2*detla_alpha))
    print("24 FDM", (tt_upper24 - tt_below24)/(2*detla_alpha))



    print("JAX AD", r0)
    print("Running 1 AD:", end_time_AD - start_time_AD)
    
    
    

if __name__ == "__main__":
    problem()
    
