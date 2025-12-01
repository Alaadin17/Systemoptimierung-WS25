import pyomo.environ as pyo

# Modell
model = pyo.ConcreteModel()

# === Parameter (gegeben) ===
model.P_PV = pyo.Param(initialize=5)        # kWh
model.P_Load = pyo.Param(initialize=8)      # kWh
model.c_grid = pyo.Param(initialize=0.3)    # €/kWh
model.P_Bat_max = pyo.Param(initialize=4)   # kWh

# === Variablen ===
model.P_Grid = pyo.Var(within=pyo.NonNegativeReals)
model.P_Bat_ch = pyo.Var(bounds=(0, model.P_Bat_max))
model.P_Bat_dis = pyo.Var(bounds=(0, model.P_Bat_max))

# === Zielfunktion ===
def cost_rule(m):
    return m.c_grid * m.P_Grid
model.Cost = pyo.Objective(rule=cost_rule, sense=pyo.minimize)

# === Energiebilanz ===
def energy_balance_rule(m):
    return m.P_PV + m.P_Grid + m.P_Bat_dis == m.P_Load + m.P_Bat_ch
model.energy_balance = pyo.Constraint(rule=energy_balance_rule)

# === Solver starten ===
solver = pyo.SolverFactory('cbc')
solver.solve(model)

# === Ergebnisse anzeigen ===
print(f"Grid power: {model.P_Grid():.2f} kWh")
print(f"Battery charge: {model.P_Bat_ch():.2f} kWh")
print(f"Battery discharge: {model.P_Bat_dis():.2f} kWh")
print(f"Total cost: {pyo.value(model.Cost):.2f} €")
