from gurobipy import GRB
import gurobipy as gp
import numpy as np
import sys


def optimize_by_gurobi(instance, timeLimit=None, export_model=False, log_output=False):

    """
    In addition to the classic gurobi optimize function, this first fixed variables
    associated with the untouchable arcs. 
    """

    # get instance parameters
    incidence_mat, commodity_dict, alpha, beta, theta = instance.get()
    max_demand = np.max(np.array([commodity_dict[c][2] for c in commodity_dict]))
    tot_demand = np.sum(np.array([commodity_dict[c][2] for c in commodity_dict]))

    # get sets (ranges)
    nodes, arcs = range(incidence_mat.shape[0]), range(incidence_mat.shape[1])
    commodities = range(len(commodity_dict))
    
    # model
    model = gp.Model("restricted network toll pricing problem")
    if log_output==False:
        model.setParam('OutputFlag', 0)

    if timeLimit is not None:
        model.setParam('TimeLimit', timeLimit)  # set time limit if specified.

    '''
    Decision Variables
    '''
    
    # binary decision variables
    xi = model.addVars(arcs, vtype=GRB.BINARY, name="xi")

    # continuous decision variables
    x = model.addVars(arcs, commodities, name="x", lb=np.zeros((len(arcs), len(commodities))))
    f = model.addVars(arcs, name="f", lb=np.zeros(len(arcs)))
    lambda_ = model.addVars(arcs, name="lambda", lb=np.zeros(len(arcs)))
    mu_ = model.addVars(nodes, name="mu")

    # set objective function
    total_congestion = gp.quicksum(alpha[a] * f[a] * f[a] + beta[a] * f[a] for a in arcs)
    model.setObjective(total_congestion, sense=GRB.MINIMIZE)

    '''
    Constraints
    '''

    # fix xi values for intouchable arcs
    intouchability_binaries = instance.get_intouchability_binaries()
    for a, binary_ in enumerate(intouchability_binaries):
        if binary_:
            model.addConstr(xi[a] == 1.0)

    # f-x relationship
    model.addConstrs((x.sum(a, '*') == f[a] for a in arcs), "f_x_")
    
    # flow balance 
    for j in commodity_dict:
        origin, destination, demand = commodity_dict[j]
        for i in nodes:
            if i == origin:
                model.addConstr(sum(incidence_mat[i, a] * x[a, j] for a in arcs)==-demand)
            elif i == destination:
                model.addConstr(sum(incidence_mat[i, a] * x[a, j] for a in arcs)== demand)
            else:
                model.addConstr(sum(incidence_mat[i, a] * x[a, j] for a in arcs)== 0)

    # either-or constraints for CS conditions
    for a in arcs:
        lambda_rhs = theta[a] + beta[a] + alpha[a] * tot_demand
        model.addConstr(x[a, 0] <= max_demand * xi[a])
        model.addConstr(lambda_[a] <= lambda_rhs * (1 - xi[a]))

    # equilibrium conditions
    for a in arcs:
        mat_mul = gp.quicksum(mu_[i] * incidence_mat[i,a] for i in nodes)
        model.addConstr(lambda_[a] - alpha[a] * f[a] - mat_mul >= beta[a])
        model.addConstr(lambda_[a] - alpha[a] * f[a] - mat_mul <= beta[a] + theta[a])

    # Save model
    if export_model:
        model.write('gurobi_base.lp')

    # solve our MIQP by gurobi
    obj_val = None
    gap_val = None
    sol_time = None

    model.optimize()  
    status = model.Status 
    if status == GRB.OPTIMAL:
        obj_val = model.ObjVal
        gap_val = model.MIPGap
        sol_time = model.Runtime
    else:
        print('Optimization was stopped with status %d' % status)
        sys.exit(0)

    return obj_val, gap_val, sol_time
