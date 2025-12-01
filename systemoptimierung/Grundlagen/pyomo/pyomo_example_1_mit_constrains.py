from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    SolverFactory,
    minimize,
)

# 1) Modell anlegen
model = ConcreteModel()

# 2) Variablen x, y >= 0
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)

# 3) Zielfunktion: min 3x + 4y
model.obj = Objective(expr=3 * model.x + 4 * model.y, sense=minimize)

# 4) Nebenbedingungen
model.c1 = Constraint(expr=2 * model.x + model.y >= 8)
model.c2 = Constraint(expr=model.x + 2 * model.y >= 10)

# 5) Solver auswählen
solver = SolverFactory("cbc")  # oder "cbc", "gurobi", ...

# 6) Lösen
result = solver.solve(model)

# 7) Ergebnis ausgeben
print("Status:", result.solver.status)
print("Termination condition:", result.solver.termination_condition)
print(f"x* = {model.x():.4f}")
print(f"y* = {model.y():.4f}")
print(f"Zielfunktionswert = {model.obj():.4f}")
