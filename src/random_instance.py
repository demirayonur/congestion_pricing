import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random 


class Instance(object):

    def __init__(self,n, m, k, my_seed, 
                 min_demand=10, 
                 max_demand=50,
                 min_alpha=1,
                 max_alpha=2,
                 min_beta = 1.5,
                 max_beta = 2.5,
                 min_theta = 0,
                 max_theta = 100):

        '''
            Generates a problem instance given the 
            number of 
            n: nodes
            m: arcs
            k: commodities
            for a given seed value for the cases where
            we use randomness. 

            Note: all entities are counted starting from zero.
        '''
        
        # fix my seeds
        np.random.seed(my_seed)
        random.seed(my_seed)

        # random graph from network X
        self.graph = nx.gnm_random_graph(n, m, seed=my_seed, directed=True)

        # get incidence matrix
        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True).todense()

        # random commodities
        demands = np.random.randint(min_demand, max_demand, k)  # random demands for commodities that will be generated
        self.commodities = {}  # we will store our commoditiies here:  commodity key --> (from_node, to_node, demand)
        counter = 0  # keeps track of how many commodities generated until that time.
        o_d_pairs = []  # we will store OD pairs here.
        while counter < k:  
            from_node = random.choice(list(range(n)))  # from which node
            to_node = random.choice(list(range(n)))    # to which node
            candidate_edge = [from_node, to_node]      # our candidate edge
            if from_node == to_node:   # no cycles are allowed
                continue
            if candidate_edge in o_d_pairs:  # repetation in OD pairs is not allowed
                continue
            if not nx.has_path(self.graph, from_node, to_node):  # there has to be at least one path between origin to destination
                continue
            # if there is no problem from the above issues:
            o_d_pairs.append(candidate_edge)  # accept the candidate edge as an OD pair.
            self.commodities[counter] = (from_node, to_node, demands[counter])  # create the commodity
            counter += 1  # increase the counter.

        # get random parameters on the arcs
        self.alpha = [round(i,2) for i in np.random.uniform(min_alpha,max_alpha, m)]
        self.beta = [round(i,2) for i in np.random.uniform(min_beta,max_beta, m)]
        self.theta = [round(i,2) for i in np.random.uniform(min_theta, max_theta, m)]
    
    def is_untouchable_arc(self, the_arc):

        '''
        It returns true if the given arc is an
        untouchable arc; returns false if not.
        '''

        Dcopy = self.graph.copy()
        Dcopy.remove_edge(the_arc[0], the_arc[1])
        flag = False
        for commodity in self.commodities:
            origin, destination = self.commodities[commodity][0:-1]
            if not nx.has_path(Dcopy, origin, destination):
                flag = True
                break
        return flag
    
    def get_intouchability_binaries(self):

        binaries = []
        for arc in self.graph.edges:
            if self.is_untouchable_arc(arc):
                binaries.append(1)
            else:
                binaries.append(0)
        return binaries

    def get(self):

        '''
        returns respectively
        --> the incidence matrix of the graph
        --> commodities dictionary
        --> alpha_values
        --> beta_values
        --> theta_valuees
        '''

        return self.incidence_matrix, self.commodities, self.alpha, self.beta, self.theta

    def get_graph(self):
        
        '''
        returns the network X graph object
        '''

        return self.graph
    
    def draw_graph(self):

        '''
        draws the defined graph in the class object
        '''

        binaries = self.get_intouchability_binaries()
        edge_colors = ["red" if i==1 else "black" for i in binaries]
        nx.draw(self.graph, cmap=plt.get_cmap('viridis'), font_color='white', edge_color=edge_colors, with_labels=True)
        plt.show()
        #nx.draw(self.graph)
        #nx.draw(self.graph, pos=nx.spring_layout(self.graph))  # use spring layout
