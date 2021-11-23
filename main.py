from random import random
import math
import numpy as np
import pandas as pd
import time as t
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from powerflow import *
from optimisation import *

#Next: add consumptions and productions as C_MW and P_MW


def random_nodes(node_count, ared_width, area_height):
    nodes = []
    pos = {}
    n = 0
    while len(nodes) < node_count:
        x, y = round(random()*ared_width, 3), round(random()*area_height, 3)
        if (x, y) not in pos:
            nodes.append([n, math.sqrt(x**2 + y**2), math.atan2(y,x)])
            pos[(x, y)] = n
            n += 1
    pos = {n: coord for coord, n in pos.items()}
    return nodes, pos

def calculate_distance(d1, d2, r1, r2):
    #Uses cosine formula to calculate direct distances between nodes
    radius = min(abs(r1-r2), 2 * math.pi - abs(r1-r2))
    return math.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * math.cos(radius))

def direct_distances(nodes, node_count):
    W = np.zeros((node_count, node_count))
    for i, d1, r1  in nodes:
        for j, d2, r2 in nodes[i+1:]:
            W[i,j] = W[j][i] =  round(calculate_distance(d1, d2, r1, r2), 1)
    return W
    
def capacity_factors(node_count, prod_count, area_height, pos):
    F = {}
    for i in range(0, prod_count):
        location_factor = 0
        if pos[i]:
           x, y = pos[i]
           #location_factor = y / area_height * 0.1 #Variance between 0 and 0.1
        F[i] = np.random.uniform(0.35, 0.45) + location_factor #Random variance between 0.35 and 0.45
    return F


    #Also try dependance on positioning
    return 

def consumptions(node_count, prod_count, total_power_level):
    C_MW = {}
    power_level = total_power_level / (node_count - prod_count)
    for i in range(prod_count, node_count):
        C_MW[i] = power_level
    return C_MW

#Do prims algorithm and store distances and paths between all nodes

def calculate_distances_edges_paths(W, D, E, P, ones, d_max = np.inf):
    maxw = np.max(W)
    #Update the new node with old node info + ([new, old])
    #Pass the reverse info for the "mirror" location
    while np.sum(D) == np.inf:
        #Find nodes that the next edge is forbidden
            #Nodes that are already part of the grid
            #Edges where neither of the nodes are part of the grid
        forbidded_off_grid_edges = ((np.nanmax(E, axis=1) == 0).T * ones).T
        forbidded_on_grid_edges = ((np.nanmax(E, axis=1) == 1).T * ones)
        W_next = W + ((forbidded_off_grid_edges + forbidded_on_grid_edges) * maxw)
        minw = np.min(W_next)
        known_n, new_n = np.argwhere(W_next== minw)[0]
        D[new_n,:] = D[known_n,:] + min(minw, d_max) #Update the distance row to the found new node /for generators (second round) use d_max
        D[:,new_n] = D[:,known_n] + min(minw, d_max) #update the distance column to the found new node
        D[new_n][new_n] = 0 # Overwrite the zero distance
        E[known_n][new_n] = E[new_n][known_n] = E[new_n][new_n] = True
        #Update the paths from and toward the new node(
        for n in np.argwhere(D[known_n,:] < np.inf):
            n=n[0]
            if n != new_n:
                #Update the path from the new node
                P[new_n,n] = [[new_n, known_n]] + P[known_n,n]
                #Reverse the list as path toward the node
                P[n, new_n] = [[n2, n1] for n1, n2 in list(reversed(P[new_n,n]))]
                #STEPS[new_n,:] = STEPS[known_n,:] + 1
                #STEPS[:,new_n] = STEPS[:,known_n] + 1
    np.fill_diagonal(E, 0)
    return D, E, P

def load_distances_edges_paths(W, node_count, prod_count):
    load_count = node_count-prod_count
    #Select only loads from the matrice
    D = np.ones((load_count, load_count)) * np.inf #Distances betwee nodes
    E = np.zeros((load_count, load_count)) #Edges: TRUE / FALSE
    P = np.empty((load_count,load_count),dtype=object) #Paths between nodes (Matrice of lists)
    #STEPS = np.zeros((node_count, node_count))
    for i,j in np.ndindex(P.shape): #Initialize paths with lists - Matrice operations dont work with the lists in numpy unfortunately
        P[i,j] = []
    graph_start = int(random()*load_count)
    E[graph_start][graph_start] = True
    D[graph_start][graph_start] = 0
    ones = np.ones((load_count, load_count)) # Helper matrice
    return calculate_distances_edges_paths(W, D, E, P, ones)

def prod_distances_and_paths(W_all, D, E, P, node_count, prod_count):
    D_all = np.ones((node_count, node_count)) * np.inf #Distances betwee nodes
    E_all = np.zeros((node_count, node_count)) #Edges: TRUE / FALSE
    P_all = np.empty((node_count,node_count),dtype=object) #Paths between nodes (Matrice of lists)
    ones = np.ones((node_count, node_count)) # Helper matrice
    D_all[prod_count:, prod_count:] = D
    E_all[prod_count:, prod_count:] = E
    P_all[prod_count:, prod_count:] = P
    W_all[:prod_count,:prod_count] = np.max(W_all) #Make sure that generator to generator edges are not selected
    d_max = 10 #km
    return calculate_distances_edges_paths(W_all, D_all, E_all, P_all, ones, d_max)

def create_shortcuts(W, D, E, P, shortcutx):
    #Calculate Connected node
    np.fill_diagonal(D, 0)
    #print(D)
    while True:
        #Get shortest cable thatcan be built to achieve the shortcutx
        shortcut_options = D - W > W * shortcutx
        if np.sum(shortcut_options) == 0:
            break
        minW = np.min(W[np.nonzero(W*shortcut_options)])
        i0,j0 = np.argwhere(W * shortcut_options == minW)[0]
        iterable_nodes = [(i0,j0),(j0,i0)] #First value i with original distances, Second value j with "new" potentially shorter paths
        E[i0,j0] = E[j0,i0] = True
        #D[i0,j0] = D[j0,i0] = W[i0,j0]
        #Do breadth first search to update the distances and paths until it is not a shortcut
        for orig_n, shortcut_n in iterable_nodes:
            d = W[orig_n,shortcut_n]
            mask = D[orig_n,:] > (D[shortcut_n,:] + d)
            
            if sum(mask) > 0:
                #Update the distance to the node with the information provided by the shortcut node
                D[orig_n, mask] = D[mask, orig_n] = D[shortcut_n,mask] + d
                #Add child nodes of this node to iterate (orig_n becomes shortcut n)
                iterable_nodes += [(child[0], orig_n) for child in np.argwhere(E[orig_n,:] == True) if child[0] != shortcut_n] #Prevent that we wont return back toward the shortcut node
                for n in np.argwhere(mask == True):
                    n=n[0]
                    if n != orig_n:
                        P[orig_n,n] = [[orig_n, shortcut_n]] + P[shortcut_n,n]
                        P[n, orig_n] = [[n2, n1] for n1, n2 in list(reversed(P[orig_n,n]))]
    return D, E, P
def print_simple_graph(E, pos, node_count, prod_count, prod_n_selected):
    # The producer node is orange if there is production and grey if no production - https://www.colorcodehex.com/231951.html
    prod_node_colors = [(255/255, 165/255, 0) if n in prod_n_selected else (128/255, 128/255, 128/255) for n in range(0,prod_count)]
    # The consumer node color is green
    cons_node_colors = [(2/255, 100/255, 64/255)]* (node_count-prod_count)
    node_colors = prod_node_colors + cons_node_colors
    Graph = nx.from_numpy_matrix(E)
    nx.draw(Graph, pos=pos, with_labels=True, font_weight='bold', node_color=node_colors)
    plt.show()

def print_graph(E, pos, node_count, prod_count, prod_n_selected, power_dict, slackbus_node=0, area_height=1000, maxcolorbar=0):
    # The producer node is orange if there is production and grey if no production - https://www.colorcodehex.com/231951.html
    prod_node_colors = [(130/255, 0/255, 0/255) if n in prod_n_selected else (217/255, 217/255, 217/255) for n in range(0,prod_count)]
    # The consumer node color is green
    cons_node_colors = [(250/255, 230/255, 50/255)]* (node_count-prod_count)
    # The slackbus is black
    node_colors = prod_node_colors + cons_node_colors
    node_colors[slackbus_node] = (35/255, 25/255, 81/255)
    E[0,0] = 1
    Graph = nx.from_numpy_matrix(E)
    edge_powers =[power_dict.get(e, 0) for e in Graph.edges]
    nodes = nx.draw_networkx_nodes(Graph,pos,node_color=node_colors, node_size=100, label=True, edgecolors = (35/255, 25/255, 81/255))
    edges = nx.draw_networkx_edges(Graph,pos,edge_color=edge_powers,width=4, edge_cmap=plt.cm.autumn_r, edge_vmax=maxcolorbar)
    dot1 = mlines.Line2D([], [], color=(130/255, 0/255, 0/255), marker='o', linestyle='None', pickradius=100, label='Production')
    dot2 = mlines.Line2D([], [], color=(217/255, 217/255, 217/255), marker='o', linestyle='None', pickradius=100, label='Potential production')
    dot3 = mlines.Line2D([], [], color=(250/255, 230/255, 50/255), marker='o', linestyle='None', pickradius=100, label='Consumption')
    dot4 = mlines.Line2D([], [], color=(35/255, 25/255, 81/255), marker='o', linestyle='None', pickradius=100, label='Slackbus')
    clb = plt.colorbar(edges)
    clb.ax.set_ylabel('Transmitted power [MW]', rotation=270, labelpad=20, fontsize=12)
    plt.axis('off')
    ax = plt.gca()
    ax.set_ylim([-50, area_height+50])
    #plt.legend(handles=[dot1, dot2, dot3, dot4], loc='lower center', ncol=2, fontsize=14)
    plt.show()
    #Labels: https://stackoverflow.com/questions/22124793/colorbar-for-edges-in-networkx
    #colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html

def optimal_power_flow(D, E, W, C_MW, F, node_count, prod_count, selected_prods):
    g = Grid(400)
    g.create_bus_grid(node_count, prod_count, selected_prods) #Give count
    g.add_branches(E, W, selected_prods, prod_count) #Give all the edges and their distances
    g.add_production(selected_prods) #Give productions sites and their bus location (int)z
    g.add_loads(C_MW) #Give loads and their bus location (int) (some buses will be empty)
    slackbus_node = g.add_slackbus(D, prod_count) #Give Distances and add to location with smallest average
    P_loss, V_ave, MW_km, power_dict, slackbus_power = g.run_power_flow()
    return slackbus_node, P_loss, V_ave, MW_km, power_dict, slackbus_power

def optimise_distance_based_production(D, F, C_MW, max_prod, node_count, prod_count, distance_loss_factor, base_case, required_capacity):
    opt = Optimisation()
    opt.add_reward_functions(D, F, C_MW, node_count, prod_count, distance_loss_factor)
    opt.add_constraints_and_objective(node_count, prod_count, max_prod, C_MW, base_case, F, required_capacity)
    result = opt.solve_and_get_prod_positioning() #Dictionary of selected sites and powers
    if base_case:
        selected_production = result
        power_sum = sum([P for n, P in result.items()])
        capacity_sum = sum([P/F[n] for n, P in result.items()])
        
    else:
        selected_production = {n: P * F[n] for n, P in result.items()}
        capacity_sum = sum([P for n, P in result.items()])
        power_sum = sum([P * F[n] for n, P in result.items()])
    return selected_production, capacity_sum, power_sum

def remove_extra_edges(E, selected_prods, prod_count):
    E_new = np.copy(E)
    for i in set(range(0, prod_count)) - set(selected_prods):
        E_new[i,:] = E_new[:,i] = 0
    return E_new

def calculation(run_id):
    #Variablesselected_prod
    run_results = []
    synth_data_results = []
    distance_loss_factor = 20 / 100 / 1000#np.random.uniform(0, 20) / 100 / 1000 # %/1000km
    node_count = 120 #total number of nodes
    prod_count = 60 #of which productions
    shortcutx = 1.15
    area_width = 900#600#900
    area_height = 900#1350#900
    total_power_level = 15000 #MW
    max_prod = 1500 #MW
    assert max_prod * prod_count > total_power_level

    #Create the input matrices
    nodes, pos = random_nodes(node_count, area_width, area_height)
    W = direct_distances(nodes, node_count)
    W_loads = np.copy(W[prod_count:, prod_count:])
    C_MW = consumptions(node_count, prod_count, total_power_level)
    F = capacity_factors(node_count, prod_count, area_height, pos)

    #Create the grid topolopgy
    D_loads, E_loads, P_loads = load_distances_edges_paths(W_loads, node_count, prod_count)
    pure_MST_branch_count = np.sum(E_loads) / 2
    pure_MST_length = np.sum(E_loads * W_loads)
    D_loads, E_loads, P_loads = create_shortcuts(W_loads, D_loads, E_loads, P_loads, shortcutx)
    D, E, P = prod_distances_and_paths(np.copy(W), D_loads, E_loads, P_loads, node_count, prod_count)

    #Optimisation and optima power flow calculation
    required_capacity = None
    try:
        for base_case, used_distance_loss_factor in [(True, 0), (False, distance_loss_factor)]:
            #First round of optimisation sets the default and second one compares
            selected_prods, capacity_sum, power_sum = optimise_distance_based_production(D, F, C_MW, max_prod, node_count, prod_count, used_distance_loss_factor, base_case, required_capacity)
            required_capacity = capacity_sum / total_power_level
            E_selected = remove_extra_edges(E, selected_prods, prod_count)
            #print_simple_graph(E, pos, node_count, prod_count, selected_prods)
            prod_distances = np.copy(W[:prod_count,:prod_count])
            for prod in range(0, prod_count):
                if prod not in selected_prods:
                    prod_distances[prod,:] = prod_distances[:,prod] = np.nan
            slackbus_node, P_loss, V_ave, MW_km, power_dict, slackbus_power = optimal_power_flow(D, E_selected, W, C_MW, F, node_count, prod_count, selected_prods)
            run_results.append([used_distance_loss_factor*100000, capacity_sum, power_sum, P_loss, V_ave, MW_km, slackbus_power, np.nanmean(prod_distances)])
            #Data related to synthetic data comparison
            cons_count = node_count-prod_count
            selected_prod_count = len(selected_prods)
            node_per_branch = np.sum(E_selected)/2 / (cons_count + selected_prod_count)
            substation_with_load = cons_count / (cons_count + selected_prod_count)
            substation_with_generators = selected_prod_count / (cons_count + selected_prod_count)
            lines_on_min_spanning_tree = pure_MST_branch_count / (np.sum(E_selected)/2)
            total_line_lenght_per_mst = np.sum(E_selected * W) / pure_MST_length
            #Add results
            synth_data_results.append([node_per_branch, substation_with_load, substation_with_generators, lines_on_min_spanning_tree, total_line_lenght_per_mst])
            if base_case:
                maxcolorbar = max([p for _,p in power_dict.items()])
            print_graph(E_selected, pos, node_count, prod_count, selected_prods.keys(), power_dict, slackbus_node, area_height, maxcolorbar)
        return [run_id, shortcutx, area_width, area_height, total_power_level, max_prod] + synth_data_results[0] + synth_data_results[1] + run_results[0] + run_results[1]
    except Exception as e:
        print(e)


print("started")
t0 = t.time()
results = []
i = 0
f = 0
while i < 1000:
    print(i)
    result = calculation(i)
    if result:
        results.append(result)
        i += 1
    else:
        f += 1
columns_initials = ["run_id",  "shortcutx", "area_width", "area_height", "total_power_level", "max_prod"]
columns_synths = ["node_per_branch", "substation_with_load", "substation_with_generators", "lines_on_min_spanning_tree", "total_line_lenght_per_mst"]
columns_results = ["used_sanction", "Sum_Capacity", "Sum_Power", "P_loss", "V_ave", "MW_km", "Slackbus_P", "Ave_Prod_Dist"]
columns_synth_base_case = [col+"_0" for col in columns_synths]
columns_results_base_case = [col+"_0" for col in columns_results]
results = pd.DataFrame(results, columns=columns_initials+columns_synth_base_case+columns_synths+columns_results_base_case+columns_results)
results["Slackbus_Need_%"] = (results["Slackbus_P"]-results["Slackbus_P_0"]) / results["Slackbus_P_0"] * 100
results["Loss_%"] = (results["P_loss"]-results["P_loss_0"]) / results["P_loss_0"] * 100
results["Power_output_%"] = (results["Sum_Power"]-results["Sum_Power_0"]) / results["Sum_Power_0"] * 100
results["Voltage_change"] = (results["V_ave"]-results["V_ave_0"]) * 400
results["MW_km_change_%"] = (results["MW_km"]-results["MW_km_0"]) / results["MW_km_0"] * 100
results["Loss%_in_base_case"] = results["P_loss_0"] / results["total_power_level"] * 100
results["Distance_change%"] = (results["Ave_Prod_Dist"] - results["Ave_Prod_Dist_0"]) / results["Ave_Prod_Dist_0"] * 100
results.to_csv("Result_3_solar_square1000.csv", sep=";",decimal=",")
print(t.time() - t0)
print("Failure percent:{}".format(f/(f+i)))
