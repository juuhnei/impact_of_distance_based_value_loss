import pandapower as pp
import pandapower.plotting
import numpy as np
import time as t

class Grid():

    def __init__(self, kV_level):
        self.network = pp.create_empty_network(f_hz=50.0)
        self.kV_level = kV_level
        self.branches = {}

    def create_bus_grid(self, node_count, prod_count, selected_prod):
        #Returns list of bus indexes
        self.buses = []
        for i in range(node_count):
            if i >= prod_count or i in selected_prod:
                self.buses.append(pp.create_bus(self.network, vn_kv = self.kV_level))
            else:
                self.buses.append(None)
        pass

    
    def add_branches(self, E, W, selected_prod, prod_count):
        branch_id = 0
        for i,j in np.argwhere(np.triu(E, k=1) == 1): #Select only upper half of the matrix 
            self.branches[branch_id] = (i,j)
            branch_id += 1
            pp.create_line_from_parameters(self.network, from_bus=self.buses[i], 
                                            to_bus=self.buses[j], length_km=W[i,j], 
                                            r_ohm_per_km=0.0171, x_ohm_per_km=0.291, #r_ohm_per_km=0.01, x_ohm_per_km=0.005, 
                                            c_nf_per_km=10, max_i_ka=0.4, 
                                            max_loading_percent=100, parallel=4,
                                            g_us_per_km=0.02)

    def add_production(self, selected_prod):
        for node, power in selected_prod.items():
            #Production site id is equals to bus id
            pp.create_gen(self.network, self.buses[node], p_mw = power, vm_pu = 1.0)

    def add_loads(self, E_MW):
        for node, power in E_MW.items():
            pp.create_load(self.network, self.buses[node], p_mw = power, q_mvar=0.0)

    def add_slackbus(self, D, prod_count):
        #Add slackbus to the centrum on consumption nodes (Average distance to rest is smallest)
        node = np.argmin(D[prod_count:,:].mean(axis=1)) + prod_count
        print("slackbus: ", node)
        pp.create_ext_grid(self.network, bus=self.buses[node], vm_pu=1.0, name="External grid")
        return node
    
    def run_power_flow(self):
        start = t.time()
        print("started to calculate power flow: ")
        try:
            pp.runpp(self.network, algorithm='nr', recycle=None, check_connectivity=True, switch_rx_ratio=2, trafo3w_losses='hv', tolerance_mva=1e-8, calculate_voltage_angles=True)
        except:
            print(pp.diagnostic(self.network))
        print("ended: ", t.time() - start)
        P_loss = sum(self.network.res_line["pl_mw"])
        V_ave = self.network.res_line[["vm_to_pu","vm_from_pu"]].mean().mean()
        MW_km = sum((abs(self.network.res_line["p_from_mw"]) + abs(self.network.res_line["p_to_mw"])) / 2 * self.network.line["length_km"])
        power_results_dict = (self.network.res_line[["p_from_mw","p_to_mw"]]).max(axis=1).to_dict()
        power_dict = {self.branches[branch_id]: int(power) for branch_id, power in power_results_dict.items()}
        power_from_slackbus = self.network.res_ext_grid["p_mw"][0]
        return P_loss, V_ave, MW_km, power_dict, power_from_slackbus

    #Run once and see how much slackbus needs to support. 

def operations():
    #NETWORK
    net1 = pp.create_empty_network(f_hz=50.0)

    #BUSES
    bus1 = pp.create_bus(net1, vn_kv = 400)
    bus2 = pp.create_bus(net1, vn_kv = 400)
    bus3 = pp.create_bus(net1, vn_kv = 400)

    #GENERATORS / LOADS / GRIDS
    pp.create_gen(net1, bus1, p_mw = 300, vm_pu = 1.0)
    pp.create_load(net1, bus2, p_mw = 300, q_mvar=0.0, in_service = True, const_z_percent = 0, const_i_percent = 0)
    pp.create_ext_grid(net1, bus=bus3, vm_pu=1.0, name="Grid Connection")

    #BRANCHES
    pp.create_line_from_parameters(net1, from_bus=bus1, to_bus=bus2, length_km=300, r_ohm_per_km=0.1, x_ohm_per_km=0.05, c_nf_per_km=0, max_i_ka=1000, max_loading_percent=100)   
    pp.create_line_from_parameters(net1, from_bus=bus2, to_bus=bus3, length_km=300, r_ohm_per_km=0.1, x_ohm_per_km=0.05, c_nf_per_km=0, max_i_ka=1000, max_loading_percent=100)   

    #print(pp.diagnostic(net1))
    #pp.plotting.simple_plot(net1)#, library = 'networkx')
    pp.runpp(net1, recycle=None, check_connectivity=True, switch_rx_ratio=2, trafo3w_losses='hv', tolerance_mva=1e-8)
    print(net1.res_line) #.gen .load