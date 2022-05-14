from docplex.mp.model import Model
import numpy as np
import sys

def optimize_by_cplex(instance, timeLimit=None, export_model=False, log_output=False):

    # get instance parameters
    incidence_mat, commodity_dict, alpha, beta, theta = instance.get()
    max_demand = np.max(np.array([commodity_dict[c][2] for c in commodity_dict]))
    tot_demand = np.sum(np.array([commodity_dict[c][2] for c in commodity_dict]))

    # get sets (ranges)
    n_node, n_arc = incidence_mat.shape
    n_commodity = len(commodity_dict)
    
    # model
    model = Model('restricted network toll pricing problem', log_output = log_output)
    if timeLimit is not None:
        model.set_time_limit(timeLimit)  # set time limit if specified.

    '''
    Decision Variables
    '''
    
    # binary decision variables
    xi = model.binary_var_list(n_arc, name='xi')

    # continuous decision variables
    x = model.continuous_var_matrix(n_arc, n_commodity, name='x', lb=0)
    f = model.continuous_var_list(n_arc, name='f', lb=0)
    lambda_ = model.continuous_var_list(n_arc, name='lambda', lb=0)       
    mu_ = model.continuous_var_list(n_node, name='mu') 

    # set objective function
    total_congestion = sum(alpha[a] * f[a] * f[a] + beta[a] * f[a] for a in range(n_arc))
    model.set_objective('min', total_congestion)

    '''
    Constraints
    '''

    # f-x relationship
    for a in range(n_arc):
        model.add_constraint(sum(x[a,j] for j in range(n_commodity)) == f[a])
    
    # flow balance 
    for j in range(n_commodity):
        origin, destination, demand = commodity_dict[j]
        for i in range(n_node):
            if i == origin:
                model.add_constraint(sum(incidence_mat[i,a] * x[a,j] for a in range(n_arc)) == -demand)
            elif i == destination:
                model.add_constraint(sum(incidence_mat[i,a] * x[a,j] for a in range(n_arc)) == demand)
            else:
                model.add_constraint(sum(incidence_mat[i,a] * x[a,j] for a in range(n_arc)) == 0)

    # equilibrium conditions
    for a in range(n_arc):
        model.add_constraint(lambda_[a] - alpha[a] * f[a] - sum(mu_[i] * incidence_mat[i,a] for i in range(n_node)) >= beta[a])
        model.add_constraint(lambda_[a] - alpha[a] * f[a] - sum(mu_[i] * incidence_mat[i,a] for i in range(n_node)) <= theta[a] + beta[a])

    # either-or constraints for CS conditions
    for a in range(n_arc):        
        lambda_rhs = theta[a] + beta[a] + alpha[a] * tot_demand
        model.add_constraint(x[a,0] <= max_demand * xi[a])
        model.add_constraint(lambda_[a] <=  lambda_rhs * (1 - xi[a]))

    # Save model
    if export_model:
        model.export_as_lp('cplex_base.lp')

    # solve our MIQP by gurobi
    obj_val = None

    s= model.solve()
    if s:
        obj_val = model.solution.objective_value
    else:
        print('Optimization was stopped')
        sys.exit(0)

    return obj_val