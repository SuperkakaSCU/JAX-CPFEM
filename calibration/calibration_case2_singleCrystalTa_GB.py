### Ref: https://doi.org/10.1115/1.4070536
### Hu: An efficient GPU-accelerated calibration of crystal plasticity model parameters by multi-objective optimization 
### with automatic differentiation‐based sensitivities


### Hu: Case2: single crystal tantalum (BCC) under tensile loading -- gradient-based calibration
### Hu: 6 Parameters: [slip_resistance_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt
import meshio
import time

## Hu: For gradient-based optimization
import scipy
from scipy.optimize import minimize

from jax_fem.solver import solver
from jax_fem.solver import implicit_vjp
#from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol
from jax_fem import logger

from applications.calibration_paper.case2 import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

case_name = 'calibration_case2'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')

alpha_dir = os.path.join(csv_dir, f'alpha.txt')
obj_dir = os.path.join(csv_dir, f'obj_val.txt')
iteration_dir = os.path.join(csv_dir, f'iteration.txt')
ss_dir = os.path.join(csv_dir, f'ss_curve_e-2.txt')



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



def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        initial_guess = problem.initial_guess if hasattr(problem, 'initial_guess') else None
        sol_list = solver(problem, {'jax_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-7, 'line_search_flag': True})
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

    ### Hu: Single crystal tantalum
    ele_type = 'HEX8'
    Nx, Ny, Nz = 1, 1, 1
    Lx, Ly, Lz = 1., 1., 1.

    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)


    disps = np.linspace(0., -0.10*Lx, 41)
    ts = np.linspace(0., 10.0, 41)



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

    quat = onp.array([[1, 0., 0., 0.]])
    cell_ori_inds = onp.zeros(len(mesh.cells), dtype=onp.int32)
    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                                additional_info=(quat, cell_ori_inds))
    

    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
    fwd_pred = ad_wrapper(problem)


    stress_curve = onp.loadtxt(ss_dir)
    stress_ref = stress_curve[:len(ts)-1]
    print(stress_ref)

    yield_stress = stress_ref[-1]
    print("yield_stress", yield_stress)
    
    def simulation(alpha):
        print("**************")
        print("alpha=",alpha)
        coeff1, coeff2, coeff3, coeff4, coeff5, coeff6 = alpha
        

        ## self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
        params = problem.internal_vars

        ## Hu: coeff1 is used to calibrate the initial slip rate
        params[1] = coeff1*params[1]
        ## Hu: coeff2 is used to calibrate the hardening parameter in K's model -- self.gss_a
        params[-5] = coeff2*params[-5]
        ## Hu: coeff3 is used to calibrate the hardening parameter in K's model -- self.h_gp
        params[-4] = coeff3*params[-4]  
        ## Hu: coeff4 is used to calibrate the saturation slip resistance -- self.t_sat_gp
        params[-3] = coeff4*params[-3]   
        ## Hu: coeff5 is used to calibrate the rate sensitivity exponent -- self.xm
        params[-2] = coeff5*params[-2]
        ## Hu: coeff6 is used to calibrate the r_gp
        params[-1] = coeff6*params[-1]

        obj_func1 = 0.
        obj_func2 = 0.
        stress_plot = np.array([])

        #point_index = np.array([0, 4, 8, 12, 16])
        point_index = onp.arange(0, len(ts)-1, 4)
        point_index = point_index.astype(int)

        for i in range(len(ts)-1):
            problem.dt = ts[i + 1] - ts[i]
            print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")
            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
            problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)

            sol_list = fwd_pred(params)
            sol = sol_list[0]

            stress_zz = problem.compute_avg_stress(sol, params)[0, 2, 2]

            params = problem.update_int_vars_gp(sol, params)

            stress_plot = np.append(stress_plot, stress_zz)
            print(f"stress_zz = {stress_zz}")

            #obj_func1 = obj_func1 + (stress_ref[i] - stress_plot[i])**2.0
        
        
        w1 = 1e6
        obj_func1 = np.sum((stress_plot-stress_ref)**2.0)
        obj1 = obj_func1/np.sum(stress_ref**2.0)

        '''
        w2 = 1e6
        obj2 = (yield_stress - stress_zz)**2.0/(stress_zz**2.0)
        '''

        return obj1*w1


    ## Hu: initialize the txt file
    alpha_write = open(alpha_dir, "w")
    alpha_write.close()

    obj_write = open(obj_dir, "w")
    obj_write.close()

    iteration_write = open(iteration_dir, "w")
    iteration_write.close()
    
    ### Hu: This new wrapper uses value_and_grad to increase efficiency
    ## Hu: Transfer jax.numpy to numpy -- objective function
    def objective_wrapper(x):
        print("***Calling objective_wrapper***")
        print(f"x = {x}")
        x = np.array(x)
        
        sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
        problem.set_initial_guess(sol_list)

        ## Hu: initialize the global variables stored by JAX between calling forward CPFEM
        problem.custom_init(quat, cell_ori_inds)

        obj_val, dJ = jax.value_and_grad(simulation)(x)
        objective_wrapper.dJ = dJ
        print(f"Finishes objective, obj_val = {obj_val}")

        obj_val = onp.array(obj_val)


        ## Hu: writing obj_val, time and alpha into file
        print("**Writing alpha into files**")
        alpha_write = open(alpha_dir, "a+")
        alpha_write.write(str(x))
        alpha_write.write('\n')
        alpha_write.close()

        print("**Writing obj value into files**")
        obj_write = open(obj_dir, "a+")
        obj_write.write(str(obj_val))
        obj_write.write('\n')
        obj_write.close()

        print("**Writing current time into files**")
        end_time_BFGS = time.time()
        run_time_BFGS = end_time_BFGS - start_time_BFGS
        iteration_write = open(iteration_dir, "a+")
        iteration_write.write(str(run_time_BFGS))
        iteration_write.write('\n')
        iteration_write.close()
        

        if obj_val < 1.0:
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            print("***Finishing Calibration***")
            raise SystemExit(0)

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

        grads = objective_wrapper.dJ

        print("***Finishing derivative***")
        
        return onp.array(grads, order='F', dtype=onp.float64)


    ## Hu: define callback for minimize()
    def callback(x):
        # callback to terminate if desired_iteration is reached
        callback.nit += 1
        desired_iteration = 100
    
    
        if callback.nit == desired_iteration:
            print("Final iterations: ", callback.nit)
            print("Final solution: ", x)
            end_time_BFGS = time.time()
            run_time_BFGS = end_time_BFGS - start_time_BFGS
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            raise StopIteration
            
        else:
            # you could print elapsed iterations, current solution
            # and current function value
            print("Elapsed iterations: ", callback.nit)
            print("Current solution: ", x)
            # print("Current function value: ", callback.fun(x))
    
    callback.nit = 0
    
    
    ## Hu: define options for minimize()
    options = {'maxiter':10, 'disp':True}

    ## Hu: define the starting point
    #pt = onp.array([1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
    pt = onp.array([2.2, 2.2, 2.2, 2.2, 2.2, 2.2])

    start_time_BFGS = time.time()
    
    
    ## Hu: perform the 'L-BFGS-B' algorithm search
    #print("***Perform the 'L-BFGS-B' algorithm search***")
    from scipy.optimize import Bounds
    print("***Perform the 'L-BFGS-B' algorithm search***")
    bounds = Bounds((0.6, 0.6, 0.6, 0.6, 0.6, 0.6), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0))
    alpha_result = minimize(objective_wrapper, pt, method='L-BFGS-B', bounds=bounds, jac=derivative_wrapper, callback = callback)



if __name__ == "__main__":
    problem()
