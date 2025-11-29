### Ref: https://doi.org/10.1115/1.4070536
### Hu: An efficient GPU-accelerated calibration of crystal plasticity model parameters by multi-objective optimization 
### with automatic differentiation‐based sensitivities


### Hu: Case1: single crystal copper (FCC) under tensile loading -- gradient-based calibration
### Hu: 6 Parameters: [slip_resistance_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt
import time


## Hu: gradient-based optimizer
## See https://scipy.org/ for more information
import scipy
from scipy.optimize import minimize


from jax_fem.solver import solver
from jax_fem.solver import implicit_vjp
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol
from jax_fem import logger


from applications.calibration.case1 import CrystalPlasticity


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
case_name = 'calibration_case1'


data_dir = os.path.join(os.path.dirname(__file__), 'data')
numpy_dir = os.path.join(data_dir, f'numpy/{case_name}')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
csv_dir = os.path.join(data_dir, f'csv/{case_name}')


## Hu: Record calibration information during optimization
alpha_dir = os.path.join(csv_dir, f'alpha.txt')
obj_dir = os.path.join(csv_dir, f'obj_val.txt')
iteration_dir = os.path.join(csv_dir, f'iteration.txt')

## Hu: Reference/Experimental data
ss_dir = os.path.join(csv_dir, f'stress_strain_curve_copper.txt')



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
        ## Hu: line search method could help reach convergence easier
        sol_list = solver(problem, {'jax_solver':{}, 'initial_guess': initial_guess, 'tol': 1e-7, 'line_search_flag': False})
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

    ### Hu: Single crystal copper represented by one element
    ele_type = 'HEX8'
    Nx, Ny, Nz = 1, 1, 1
    Lx, Ly, Lz = 1., 1., 1.

    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    ## Hu: Rotation applied on the single crystal
    quat = onp.array([[1, 0., 0., 0.]])
    cell_ori_inds = onp.zeros(len(mesh.cells), dtype=onp.int32)

    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)


    ## Hu: Displacement and time conditions
    disps = np.linspace(0., 0.025*Lx, 21)
    ts = np.linspace(0., 2.5, 21)


    ## Hu: Define index of points and faces
    def corner(point):
        flag_x = np.isclose(point[0], 0., atol=1e-5)
        flag_y = np.isclose(point[1], 0., atol=1e-5)
        flag_z = np.isclose(point[2], 0., atol=1e-5)
        return np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

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


    ## Hu: Define CPFEM problem on top of JAX-FEM
    ## Xue, Tianju, et al. Computer Physics Communications 291 (2023): 108802.
    problem = CrystalPlasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                                additional_info=(quat, cell_ori_inds))
    

    

    sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
    fwd_pred = ad_wrapper(problem)


    ## Hu: Read Reference/Experimental stress-strain curve
    ## Reference curve calculated based on reference material parameters
    stress_curve = onp.loadtxt(ss_dir)
    stress_ref = stress_curve[:len(ts)-1]
    print("Reference stress-strain curve", stress_ref)


    yield_stress = stress_ref[-1]
    print("Yield stress", yield_stress)



    def simulation(alpha):
        print("**************")
        print("Parameter scaling optimization: alpha=",alpha)
        coeff1, coeff2, coeff3, coeff4, coeff5, coeff6 = alpha
        

        ## Hu: Key part of this code: Do calibration directly on the scaling coefficients of material parameters on each Gauss point
        ## Hu: self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp, gss_a_gp, h_gp, t_sat_gp, xm_gp, r_gp]
        params = problem.internal_vars

        ## Hu: Apply coefficients directly on the material parameters on each Gauss point
        ## Hu: coeff1 is used to calibrate the initial slip rate -- self.slip_resistance_gp
        params[1] = coeff1*params[1]
        ## Hu: coeff2 is used to calibrate the hardening parameter in K's model -- self.gss_a
        params[-5] = coeff2*params[-5]
        ## Hu: coeff3 is used to calibrate the hardening parameter in K's model -- self.h_gp
        params[-4] = coeff3*params[-4]  
        ## Hu: coeff4 is used to calibrate the saturation slip resistance -- self.t_sat_gp
        params[-3] = coeff4*params[-3]   
        ## Hu: coeff5 is used to calibrate the rate sensitivity exponent -- self.xm
        params[-2] = coeff5*params[-2]
        ## Hu: coeff6 is used to calibrate the self.r_gp
        params[-1] = coeff6*params[-1]


        ## Hu: Objective Function
        obj_func1 = 0.
        obj_func2 = 0.
        stress_plot = np.array([])


        ## Hu: Definition of reference points si
        point_index = onp.arange(0, len(ts)-1, 4)
        point_index = point_index.astype(int)


        for i in range(len(ts)-1):
            problem.dt = ts[i + 1] - ts[i]
            print(f"\nStep {i + 1} in {len(ts) - 1}, disp = {disps[i + 1]}, dt = {problem.dt}")
            dirichlet_bc_info[-1][-1] = get_dirichlet_top(disps[i + 1])
            problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)


            ## Hu: Get f_simulated based on updated alpha
            sol_list = fwd_pred(params)
            sol = sol_list[0]

            
            stress_zz = problem.compute_avg_stress(sol, params)[0, 2, 2]

            ## Hu: Update internal variables
            params = problem.update_int_vars_gp(sol, params)

            ## Summarize f_simulated
            stress_plot = np.append(stress_plot, stress_zz)
            print(f"stress_zz = {stress_zz}")

            
            
        ### Hu: You can choose different objective function for calibration
        ## Study 0: Only consider yield point -- not appropriate
        #obj = (yield_stress - stress_zz)**2.0

        
        ## Study 1: difference between the experimentally determined stress and the simulated stress at selected strain values
        w1 = 1e6
        obj_func1 = np.sum((stress_plot[point_index]-stress_ref[point_index])**2.0)
        obj1 = obj_func1/np.sum(stress_ref[point_index]**2.0)

        print("!!!!Hey here!!!!")
        return obj1*w1
        

        '''
        ## Study 2: d2: penalize a difference in the slope of the curves
        w2 = 1e6
        obj2 = ((stress_plot[-1]-stress_plot[4])-(stress_ref[-1]-stress_ref[4]))/(stress_ref[-1]-stress_ref[4])
        obj2 = np.absolute(obj2)

        return obj2*w2
        '''

        '''
        ## Study 3: reinforced objective function
        w1 = 1e6
        obj_func1 = np.sum((stress_plot-stress_ref)**2.0)
        obj1 = obj_func1/np.sum(stress_ref**2.0)


        w2 = 1e6
        obj2 = (yield_stress - stress_zz)**2.0/(stress_zz**2.0)

        obj = obj1*w1 + obj2*w2
        print("obj:{0}=obj1:{1}+obj2:{2}".format(obj,obj1,obj2))
        '''


        
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
        #print(f"\nobjective_wrapper: Type of x = {type(x)}")
        print(f"x = {x}")
        x = np.array(x)
        #print(f"objective_wrapper: Type of x = {type(x)}")

        ## Hu: writing obj_val, time and alpha into file
        print("**Writing alpha into files**")
        alpha_write = open(alpha_dir, "a+")
        #alpha_write.write(str(x))
        alpha_write.write(" ".join(map(str, x)))
        alpha_write.write('\n')
        alpha_write.close()


        sol_list = [np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))]
        problem.set_initial_guess(sol_list)
        ## Hu: initialize the global variables stored by JAX between calling forward CPFEM
        problem.custom_init(quat, cell_ori_inds)

        obj_val, dJ = jax.value_and_grad(simulation)(x)
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
        

        
        if obj_val < 1.0:
            print("The running time(sec) for BFGS calibration is: ", run_time_BFGS)
            print("***Finishing Calibration***")
            #warnings.warn("Terminating optimization: iteration limit reached",TookTooManyIters)
            raise SystemExit(0)

        #print(f"Type of obj_val = {type(obj_val)}")
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
        #print(f"grads.shape = {grads.shape}")

        # 'L-BFGS-B' & 'BFGS' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(grads, order='F', dtype=onp.float64)
    
   

    ## Hu: define callback for minimize()
    def callback(x):
        # callback to terminate if desired_iteration is reached
        callback.nit += 1
        desired_iteration = 10 # for example you want it to stop after 10 iterations
    
        #if callback.fun(x) < 8.5 or callback.nit == desired_iteration:
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
            # print("Current function value: ", callback.fun(x))
    
    callback.nit = 0
    
    
    ## Hu: define options for minimize()
    options = {'maxiter':10, 'disp':True}

    ## Hu: define the starting point
    pt = onp.array([1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
    #pt = onp.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    start_time_BFGS = time.time()
    
    ### Hu: different gradient-based algorithm search, you can have a try of CG, BFGS, and so on.
    ## Hu: See Scipy website for details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    from scipy.optimize import Bounds
    print("***Perform the 'L-BFGS-B' algorithm search***")
    bounds = Bounds((0.4, 0.4, 0.4, 0.4, 0.4, 0.4), (3.0, 3.0, 3.0, 3.0, 3.0, 3.0))
    alpha_result = minimize(objective_wrapper, pt, method='L-BFGS-B', bounds=bounds, jac=derivative_wrapper, callback = callback)
    

## Hu: post-processing treatment
def plot_result_summary():
    alpha_GB = onp.loadtxt(alpha_dir)
    iteration_GB = onp.loadtxt(iteration_dir)
    objective_GB = onp.loadtxt(obj_dir)

    print("The reference scaling parameters are: [1., 1., 1., 1., 1., 1.]")
    print("The final optimized scaling parameters are: {0}, taking {1:.1f} seconds".format(alpha_GB[-1,:], iteration_GB[-1]))
    print("The final objective value is: {0:.1f}, and it takes {1} iterations".format(objective_GB[-1], objective_GB.shape[0]))

    


if __name__ == "__main__":
    problem()
    plot_result_summary()
    
