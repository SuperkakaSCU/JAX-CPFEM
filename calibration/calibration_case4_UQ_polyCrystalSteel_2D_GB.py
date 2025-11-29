### Ref: https://doi.org/10.1115/1.4070536
### Hu: An efficient GPU-accelerated calibration of crystal plasticity model parameters by multi-objective optimization 
### with automatic differentiation‐based sensitivities

### Hu: Case4: polycrystal 304steel under 2D loading -- gradient-based calibration
### Hu: 6 Parameters: [slip_resistance_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt
import time
import meshio


## Hu: For gradient-based optimization
import scipy
from scipy.optimize import minimize


from jax_fem.solver import solver
from jax_fem.solver import implicit_vjp
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol
from jax_fem import logger


from applications.calibration_paper.case4 import CrystalPlasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

case_name = 'calibration_case4'

data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')
neper_folder = os.path.join(data_dir, f'neper/{case_name}')

ori_dir = os.path.join(csv_dir, f'grain_inds.txt')
ss_dir = os.path.join(csv_dir, f'UQ/stress_zz_curve_scenario0.txt')
ss1_dir = os.path.join(csv_dir, f'UQ/stress_xx_curve_scenario1.txt')


alpha_dir = os.path.join(csv_dir, f'UQ/alpha_2D.txt')
obj_dir = os.path.join(csv_dir, f'UQ/obj_val_2D.txt')
iteration_dir = os.path.join(csv_dir, f'UQ/iteration_2D.txt')



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
        # sol_list = solver(problem, {'jax_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-7, 'line_search_flag': True})
        sol_list = solver(problem, {'umfpack_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-7, 'line_search_flag': True})
        problem.set_initial_guess(sol_list)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        # vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options={'jax_solver':{}, 'initial_guess': sol_list, 'tol': 1e-7, 'line_search_flag': True})
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options={'umfpack_solver':{}, 'initial_guess': sol_list, 'tol': 1e-7, 'line_search_flag': True})

        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred



def problem():
    print(jax.lib.xla_bridge.get_backend().platform)


    pf_args = {}
    pf_args['data_dir'] = data_dir
    pf_args['num_grains'] = 50
    pf_args['id'] = 0
    pf_args['domain_x'] = 0.0295     ## domain size of cuboidal
    pf_args['domain_y'] = 0.0295 
    pf_args['domain_z'] = 0.0295 
    pf_args['num_oris'] = 50


    ### Hu: Single crystal copper
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = meshio.read(os.path.join(neper_folder, f"UQ/n50-id0.msh"))
    cell_grain_inds = meshio_mesh.cell_data['gmsh:physical'][0] - 1


    grain_oris_inds = onp.arange(pf_args['num_oris'])
    print("grain_oris_inds.shape",grain_oris_inds.shape)
    print(grain_oris_inds)

    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])


    quat_file = os.path.join(csv_dir, f"quat.txt")
    quat = onp.loadtxt(quat_file)[:pf_args['num_oris'], 1:]
    


    Lx = np.max(mesh.points[:, 0])
    Ly = np.max(mesh.points[:, 1])
    Lz = np.max(mesh.points[:, 2])


    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)


    disps = np.linspace(0., 0.025*Lx, 81)
    ts = np.linspace(0., 2.5, 81)

    disps1 = np.linspace(0., 0.025*Lx, 81)
    ts1 = np.linspace(0., 2.5, 81)


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

    def zero_dirichlet_val(point):
        return 0.

    def get_dirichlet_top(disp):
            def val_fn(point):
                return disp
            return val_fn

    dirichlet_bc_info = [[corner2, corner2, bottom, top], 
                         [0, 1, 2, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps[0])]]


    dirichlet_bc_info1 = [[corner2, corner2, left, right], 
                         [1, 2, 0, 0], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, get_dirichlet_top(disps1[0])]]

    
    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, 
                                dirichlet_bc_info=dirichlet_bc_info, additional_info=(quat, cell_ori_inds))


    problem1 = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, 
                                dirichlet_bc_info=dirichlet_bc_info1, additional_info=(quat, cell_ori_inds))
    

    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
    sol_list1 = [np.zeros((problem1.fes[0].num_total_nodes, problem1.fes[0].vec))]


    fwd_pred = ad_wrapper(problem)
    fwd_pred1 = ad_wrapper(problem1)

    ## Hu: Strain Rate = 0.01 s-1
    ### stress_zz for scenario0
    stress_curve = onp.loadtxt(ss_dir)
    stress_ref = stress_curve[:len(ts)-1]
    print("stress_zz for scenario0 before noise:", stress_ref)
    yield_stress = stress_ref[-1]
    print("yield stress for scenario0 before noise:", yield_stress)
    # [-3, 3]
    onp.random.seed(0)
    noise = onp.random.uniform(low=-3, high=3, size=stress_ref.shape)
    stress_ref = stress_ref + noise
    print("stress_zz for scenario0 after noise:", stress_ref)
    onp.savetxt(os.path.join(csv_dir, f"UQ/stress_zz_curve_scenario0_2D_noise.txt"), stress_ref)


    ### stress_xx for scenario1
    stress_curve1 = onp.loadtxt(ss1_dir)
    stress_ref1 = stress_curve1[:len(ts1)-1]
    print("stress_xx for scenario1", stress_ref1)
    yield_stress1 = stress_ref1[-1]
    print("yield_stress for scenario1", yield_stress1)
    # [-3, 3]
    onp.random.seed(1)
    noise1 = onp.random.uniform(low=-3, high=3, size=stress_ref1.shape)
    stress_ref1 = stress_ref1 + noise1
    print("stress_xx for scenario1 after noise:", stress_ref1)
    onp.savetxt(os.path.join(csv_dir, f"UQ/stress_xx_curve_scenario0_2D_noise.txt"), stress_ref1)
    


    def simulation(alpha):
        print("**************")
        print("alpha=",alpha)
        coeff1, coeff2, coeff3, coeff4, coeff5, coeff6 = alpha
        

        ## self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
        params = problem.internal_vars
        params1 = problem1.internal_vars


        ## Scenario 0
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


        ## Scenario 1
        ## Hu: coeff1 is used to calibrate the initial slip rate
        params1[1] = coeff1*params1[1]
        ## Hu: coeff2 is used to calibrate the hardening parameter in K's model -- self.gss_a
        params1[-5] = coeff2*params1[-5]
        ## Hu: coeff3 is used to calibrate the hardening parameter in K's model -- self.h_gp
        params1[-4] = coeff3*params1[-4]  
        ## Hu: coeff4 is used to calibrate the saturation slip resistance -- self.t_sat_gp
        params1[-3] = coeff4*params1[-3]   
        ## Hu: coeff5 is used to calibrate the rate sensitivity exponent -- self.xm
        params1[-2] = coeff5*params1[-2]
        ## Hu: coeff6 is used to calibrate the r_gp
        params1[-1] = coeff6*params1[-1]


        obj_func1 = 0.
        obj_func2 = 0.

        obj_func1_1 = 0.
        obj_func1_2 = 0.


        stress_zz_plot_scenario0 = np.array([])
        von_Mises_plot_scenario0 = np.array([])
        

        stress_xx_plot_scenario1 = np.array([])
        von_Mises_plot_scenario1 = np.array([])

        
        point_index = np.arange(0, len(ts)-1, 4)
        #point_index = point_index.astype(int)
        
        

        for i in range(len(ts)-1):
        #for i in range(5):
            problem.dt = ts[i + 1] - ts[i]
            problem1.dt = ts1[i + 1] - ts1[i]
            print(f"\nScenario0: Step {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")
            print(f"Scenario1: Step {i + 1} in {len(ts1) - 1}, disp = {disps1[i + 1]}, dt = {problem1.dt}")
            
            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
            dirichlet_bc_info1[-1][-1] = get_dirichlet_top(disps1[i + 1])

            problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
            problem1.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info1)


            print("***Solving scenario0***")
            sol_list = fwd_pred(params)
            print("***Solving scenario1***")
            sol_list1 = fwd_pred1(params1)
            
            
            sol = sol_list[0]
            sol1 = sol_list1[0]
            


            print(f"Computing stress for scenario0...")
            sigma_cell_data = jax.checkpoint(problem.compute_avg_stress)(sol, params)
            print(f"Computing stress for scenario1...")
            sigma_cell_data1 = jax.checkpoint(problem1.compute_avg_stress)(sol1, params1)


            print(f"Computing stress components for scenario0......")
            # sigma_cell_xx = sigma_cell_data[:, 0, 0]
            # sigma_cell_yy = sigma_cell_data[:, 1, 1]
            sigma_cell_zz = sigma_cell_data[:, 2, 2]
            # sigma_cell_xy = sigma_cell_data[:, 0, 1]
            # sigma_cell_xz = sigma_cell_data[:, 0, 2]
            # sigma_cell_yz = sigma_cell_data[:, 1, 2]
            
            # print(f"Computing Von Mises Stress for scenario0...")
            # sigma_cell_von_Mises_stress = (0.5*((sigma_cell_xx - sigma_cell_yy)**2.0 + (sigma_cell_yy - sigma_cell_zz)**2.0 + (sigma_cell_zz - sigma_cell_xx)**2.0) + \
            #                 + 3.0*(sigma_cell_xy**2.0 + sigma_cell_yz**2.0 + sigma_cell_xz**2.0))**0.5



            print(f"Computing stress components for scenario1......")
            sigma_cell_xx1 = sigma_cell_data1[:, 0, 0]
            # sigma_cell_yy1 = sigma_cell_data1[:, 1, 1]
            # sigma_cell_zz1 = sigma_cell_data1[:, 2, 2]
            # sigma_cell_xy1 = sigma_cell_data1[:, 0, 1]
            # sigma_cell_xz1 = sigma_cell_data1[:, 0, 2]
            # sigma_cell_yz1 = sigma_cell_data1[:, 1, 2]
            
            # print(f"Computing Von Mises Stress for scenario1...")
            # sigma_cell_von_Mises_stress1 = (0.5*((sigma_cell_xx1 - sigma_cell_yy1)**2.0 + (sigma_cell_yy1 - sigma_cell_zz1)**2.0 + (sigma_cell_zz1 - sigma_cell_xx1)**2.0) + \
            #                 + 3.0*(sigma_cell_xy1**2.0 + sigma_cell_yz1**2.0 + sigma_cell_xz1**2.0))**0.5


            params = problem.update_int_vars_gp(sol, params)
            params1 = problem1.update_int_vars_gp(sol1, params1)

            
            stress_zz_plot_scenario0 = np.append(stress_zz_plot_scenario0, np.mean(sigma_cell_zz))
            # von_Mises_plot_scenario0 = np.append(von_Mises_plot_scenario0, np.mean(sigma_cell_von_Mises_stress))


            stress_xx_plot_scenario1 = np.append(stress_xx_plot_scenario1, np.mean(sigma_cell_xx1))
            # von_Mises_plot_scenario1 = np.append(von_Mises_plot_scenario1, np.mean(sigma_cell_von_Mises_stress1))


            #obj_func1 = obj_func1 + (stress_ref[i] - stress_plot[i])**2.0
            

        
        # MSE: penalize a difference in the curves
        w1_0 = 1e6
        obj_func1_0 = np.sum((stress_zz_plot_scenario0-stress_ref)**2.0)
        obj1_0 = obj_func1_0/np.sum(stress_ref**2.0)
        
        
        w1_1 = 1e6
        obj_func1_1 = np.sum((stress_xx_plot_scenario1-stress_ref1)**2.0)
        obj1_1 = obj_func1_1/np.sum(stress_ref1**2.0)
        
        
        obj = w1_0*obj1_0 + w1_1*obj1_1
        #print("stress_zz:\n", onp.array(stress_zz_plot, order='F', dtype=onp.float64))
        print("!!!!Hey here!!!!")
        print("obj:{0}=obj1_0:{1}+obj1_1:{2}".format(obj,obj1_0,obj1_1))
        
        return obj
        
        
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

        ## Hu: writing obj_val, time and alpha into file
        print("**Writing alpha into files**")
        alpha_write = open(alpha_dir, "a+")
        alpha_write.write(str(x))
        alpha_write.write('\n')
        alpha_write.close()


        sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
        sol_list1 = [np.zeros((problem1.fes[0].num_total_nodes, problem1.fes[0].vec))]

        problem.set_initial_guess(sol_list)
        problem1.set_initial_guess(sol_list1)


        ## Hu: initialize the global variables stored by JAX between calling forward CPFEM
        problem.custom_init(quat, cell_ori_inds)
        problem1.custom_init(quat, cell_ori_inds)
        value_and_grad_fn = jax.value_and_grad(simulation)
        obj_val, dJ = value_and_grad_fn(x)
        
        objective_wrapper.dJ = dJ
        print(f"Finishes objective, obj_val = {obj_val}")
        #print(f"Type of obj_val = {type(obj_val)}")

        obj_val = onp.array(obj_val)
        

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
        
        del value_and_grad_fn
        del dJ
        
        if obj_val < 1.0:
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            print("***Finishing Calibration***")
            #warnings.warn("Terminating optimization: iteration limit reached",TookTooManyIters)
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
        sol_list1 = [np.zeros((problem1.fes[0].num_total_nodes, problem1.fes[0].vec))]


        problem.set_initial_guess(sol_list)
        problem1.set_initial_guess(sol_list1)


        problem.custom_init(quat, cell_ori_inds)
        problem1.custom_init(quat, cell_ori_inds)

        grads = objective_wrapper.dJ

        print("***Finishing derivative***")

        # 'L-BFGS-B' & 'BFGS' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(grads, order='F', dtype=onp.float64)


    
    ## Hu: define callback for minimize()
    def callback(x):
        # callback to terminate if desired_iteration is reached
        callback.nit += 1
        desired_iteration = 25 
    
        if callback.nit == desired_iteration:
            print("Final iterations: ", callback.nit)
            print("Final solution: ", x)
            #print("Current function value: ", callback.fun(x))
            end_time_BFGS = time.time()
            run_time_BFGS = end_time_BFGS - start_time_BFGS
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            raise StopIteration
            
        else:
            # you could print elapsed iterations, current solution
            # and current function value
            print("Elapsed iterations: ", callback.nit)
            print("Current solution: ", x)
    
    callback.nit = 0
    
    
    ## Hu: define options for minimize()
    options = {'maxiter':25, 'disp':True}

    ## Hu: define the starting point
    pt = onp.array([1.4, 1.4, 1.4, 1.4, 1.4, 1.4])

    start_time_BFGS = time.time()
    
    
    from scipy.optimize import Bounds
    print("***Perform the 'L-BFGS-B' algorithm search***")
    bounds = Bounds((0.7, 0.7, 0.7, 0.7, 0.7, 0.7), (3.0, 3.0, 3.0, 3.0, 3.0, 2.5))
    alpha_result = minimize(objective_wrapper, pt, method='L-BFGS-B', bounds=bounds, jac=derivative_wrapper, callback = callback)
    

    print("stress_zz for scenario0 after noise:", stress_ref)
    print("stress_xx for scenario1 after noise:", stress_ref1)



if __name__ == "__main__":
    problem()
    