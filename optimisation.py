import sys
import numpy as np
import pulp
from collections import defaultdict

class Optimisation():
    def __init__(self):
        self.model = pulp.LpProblem(name = "Prodution_location_max", sense = pulp.LpMaximize)
    
    def add_reward_functions(self, D, F, C_MW, node_count, prod_count, distance_loss_factor):
        self.rewards = {}
        for p in range(0, prod_count):
            for c in range(prod_count, node_count):
                self.rewards[(p,c)] = F[p]*C_MW[c] - D[p,c] * distance_loss_factor * C_MW[c]
                #print(D[p,c] * distance_loss_factor)

    def add_constraints_and_objective(self, node_count, prod_count, max_prod, C_MW, base_case, F, required_capacity):
        #Decision variables
        pairs = [(p, c) for p in range(0,prod_count) for c in range(prod_count, node_count)]
        self.vars = pulp.LpVariable.dicts("Production_locationing", pairs, lowBound = 0, cat = 'Continuous')
        objective = pulp.LpAffineExpression(e = [(self.vars[p,c],self.rewards[p,c]) for p,c in self.vars], name = 'Objective function')
        
        self.model += pulp.lpSum(objective)

        if base_case:
            #The required energy by the consumption sites is the limit
            # Constraint: consumption power per site = sum of production power
            for c in range(prod_count, node_count):
                tmpExpression = pulp.LpAffineExpression(e = [(self.vars[p,c], 1) for p in range(0, prod_count)])
                tmpConstraint = pulp.LpConstraint(e = pulp.lpSum(tmpExpression),
                    sense = pulp.LpConstraintEQ,                                
                    rhs = C_MW[c])
                self.model.addConstraint(tmpConstraint)
            # Constraint: power production per site <= maximum capacity * capacity_factor
            for p in range(0, prod_count):
                tmpExpression = pulp.LpAffineExpression(e = [(self.vars[p,c], 1) for c in range(prod_count, node_count)])
                tmpConstraint = pulp.LpConstraint(e = pulp.lpSum(tmpExpression),
                    sense = pulp.LpConstraintLE,
                    rhs = max_prod * F[p])
                self.model.addConstraint(tmpConstraint)
        else: 
            #Installed capacity sets the limits - since the capacity has to match to the base case
            # Constraint: required capacity per site = production capacities
            for c in range(prod_count, node_count):
                tmpExpression = pulp.LpAffineExpression(e = [(self.vars[p,c], 1) for p in range(0, prod_count)])
                tmpConstraint = pulp.LpConstraint(e = pulp.lpSum(tmpExpression),
                    sense = pulp.LpConstraintEQ,                                
                    rhs = C_MW[c] * required_capacity)
                self.model.addConstraint(tmpConstraint)
            # Constraint: production capacity per site <= maximum capacity
            for p in range(0, prod_count):
                tmpExpression = pulp.LpAffineExpression(e = [(self.vars[p,c], 1) for c in range(prod_count, node_count)])
                tmpConstraint = pulp.LpConstraint(e = pulp.lpSum(tmpExpression),
                    sense = pulp.LpConstraintLE,
                    rhs = max_prod)
                self.model.addConstraint(tmpConstraint)
        

    def solve_and_get_prod_positioning(self):
        solver = pulp.GLPK_CMD(msg=0)
        self.model.solve(solver)
        #check the results
        if self.model.status == True:
            print('Solution found: %s' %pulp.LpStatus[self.model.status])
        else:
            print('Failed to find solution: %s' %pulp.LpStatus[self.model.status])
        print('Objective function value =', pulp.value(self.model.objective))
        results = defaultdict(int)
        #sum_power = 0
        for p, c in self.vars:
            if self.vars[p,c].varValue > 0.01:
                results[p] += round(self.vars[p,c].varValue,1)
        #In the base case, solver gives production powers to meet the consumption
        #In later it gives capacities that match to the base case
        return results