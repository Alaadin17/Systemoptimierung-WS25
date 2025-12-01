from pyomo.environ import ConcreteModel, Var, Objective, Constraint, Set, NonNegativeReals, SolverFactory, minimize

# 1) Modell-Container
model = ConcreteModel()

# 2) Variablen
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)


# 3) Zielfunktion
model.obj = Objective(expr=3*model.x + 4*model.y, sense=minimize)

# 4) Constraints: direkt mit expr
model.c1 = Constraint(expr=2*model.x + model.y >= 8)
model.c2 = Constraint(expr=model.x + 2*model.y >= 10)

# 5) Lösen
solver = SolverFactory("cbc")  # oder glpk, gurobi, cplex ...
result = solver.solve(model)
print(model.x(), model.y(), model.obj())

