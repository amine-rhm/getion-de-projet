import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


class SimplexSolver:
    def __init__(self):
        self.tableau = None
        self.num_variables = 0
        self.num_constraints = 0
        self.basic_variables = []
        self.objective_function = []
        self.constraints = []
        self.b_values = []
        self.constraint_types = []
        self.min_or_max = "max"
        self.result = {}

    def define_problem(self, objective_function, constraints, b_values, constraint_types, min_or_max="max"):
        self.objective_function = objective_function
        self.constraints = constraints
        self.b_values = b_values
        self.constraint_types = constraint_types
        self.min_or_max = min_or_max
        self.num_variables = len(objective_function)
        self.num_constraints = len(constraints)

    def _prepare_tableau(self):
        num_slack = sum(1 for constr_type in self.constraint_types if constr_type == "<=")
        num_surplus = sum(1 for constr_type in self.constraint_types if constr_type == ">=")
        num_artificial = sum(1 for constr_type in self.constraint_types if constr_type in [">=", "="])

        total_variables = self.num_variables + num_slack + num_surplus + num_artificial
        self.tableau = np.zeros((self.num_constraints + 1, total_variables + 1))

        for j in range(self.num_variables):
            if self.min_or_max == "max":
                self.tableau[0, j] = -self.objective_function[j]
            else:
                self.tableau[0, j] = self.objective_function[j]

        slack_idx = self.num_variables
        surplus_idx = slack_idx + num_slack
        artificial_idx = surplus_idx + num_surplus

        self.basic_variables = []

        for i in range(self.num_constraints):
            for j in range(self.num_variables):
                self.tableau[i + 1, j] = self.constraints[i][j]
            self.tableau[i + 1, -1] = self.b_values[i]

            if self.constraint_types[i] == "<=":
                self.tableau[i + 1, slack_idx] = 1
                self.basic_variables.append(slack_idx)
                slack_idx += 1
            elif self.constraint_types[i] == ">=":
                self.tableau[i + 1, surplus_idx] = -1
                self.tableau[i + 1, artificial_idx] = 1
                self.basic_variables.append(artificial_idx)
                self.tableau[0, artificial_idx] = 1000
                surplus_idx += 1
                artificial_idx += 1
            elif self.constraint_types[i] == "=":
                self.tableau[i + 1, artificial_idx] = 1
                self.basic_variables.append(artificial_idx)
                self.tableau[0, artificial_idx] = 1000
                artificial_idx += 1

    def _is_optimal(self):
        if self.min_or_max == "max":
            return all(self.tableau[0, :-1] >= 0)
        else:
            return all(self.tableau[0, :-1] <= 0)

    def _select_entering_variable(self):
        if self.min_or_max == "max":
            return np.argmin(self.tableau[0, :-1])
        else:
            return np.argmax(self.tableau[0, :-1])

    def _select_leaving_variable(self, entering_idx):
        ratios = []
        for i in range(1, self.num_constraints + 1):
            if self.tableau[i, entering_idx] > 0:
                ratios.append((self.tableau[i, -1] / self.tableau[i, entering_idx], i))
            else:
                ratios.append((float('inf'), i))

        if not ratios or all(ratio[0] == float('inf') for ratio in ratios):
            return None

        min_ratio, leaving_row = min(ratios, key=lambda x: x[0])
        return leaving_row

    def _pivot(self, entering_idx, leaving_row):
        pivot_element = self.tableau[leaving_row, entering_idx]
        self.tableau[leaving_row] = self.tableau[leaving_row] / pivot_element

        for i in range(self.num_constraints + 1):
            if i != leaving_row:
                multiplier = self.tableau[i, entering_idx]
                self.tableau[i] = self.tableau[i] - multiplier * self.tableau[leaving_row]

        self.basic_variables[leaving_row - 1] = entering_idx

    def solve(self):
        self._prepare_tableau()
        self._phase_one()
        self._phase_two()
        self._extract_solution()
        return self.result

    def _phase_one(self):
        artificial_vars = []
        for i, var_idx in enumerate(self.basic_variables):
            if var_idx >= self.num_variables + sum(1 for ct in self.constraint_types if ct == "<=") + sum(
                    1 for ct in self.constraint_types if ct == ">="):
                artificial_vars.append((i + 1, var_idx))

        if not artificial_vars:
            return

        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if all(self.tableau[row, -1] == 0 for row, _ in artificial_vars):
                break

            entering_idx = None
            for j in range(self.tableau.shape[1] - 1):
                if j in [var_idx for _, var_idx in artificial_vars]:
                    continue

                if entering_idx is None or self.tableau[0, j] < self.tableau[0, entering_idx]:
                    entering_idx = j

            if entering_idx is None or self.tableau[0, entering_idx] >= 0:
                break

            leaving_row = self._select_leaving_variable(entering_idx)
            if leaving_row is None:
                self.result["status"] = "unbounded"
                return

            self._pivot(entering_idx, leaving_row)

        for row, var_idx in artificial_vars:
            if self.basic_variables[row - 1] == var_idx and self.tableau[row, -1] != 0:
                self.result["status"] = "infeasible"
                return

    def _phase_two(self):
        max_iterations = 100
        iteration = 0

        while not self._is_optimal() and iteration < max_iterations:
            iteration += 1

            entering_idx = self._select_entering_variable()

            if self.min_or_max == "min" and self.tableau[0, entering_idx] <= 0:
                break

            if self.min_or_max == "max" and self.tableau[0, entering_idx] >= 0:
                break

            leaving_row = self._select_leaving_variable(entering_idx)
            if leaving_row is None:
                self.result["status"] = "unbounded"
                return

            self._pivot(entering_idx, leaving_row)

    def _extract_solution(self):
        solution = np.zeros(self.num_variables)
        for i, var_idx in enumerate(self.basic_variables):
            if var_idx < self.num_variables:
                solution[var_idx] = self.tableau[i + 1, -1]

        self.result["variables"] = solution
        self.result["objective_value"] = -self.tableau[0, -1] if self.min_or_max == "max" else self.tableau[0, -1]


def solve_transportation_problem():
    print("\n=== PROBLÈME DE TRANSPORT ===")

    # Données du problème
    costs = np.array([
        [10, 12, 15, 8],  # Usine A
        [14, 10, 16, 11],  # Usine B
        [12, 9, 11, 13]  # Usine C
    ])

    supply = np.array([100, 80, 120])
    demand = np.array([70, 50, 90, 90])

    # Vérification de l'équilibre
    total_supply = np.sum(supply)
    total_demand = np.sum(demand)
    print(f"Offre totale: {total_supply}")
    print(f"Demande totale: {total_demand}")

    if total_supply != total_demand:
        if total_supply > total_demand:
            dummy_demand = total_supply - total_demand
            demand = np.append(demand, dummy_demand)
            costs = np.hstack((costs, np.zeros((costs.shape[0], 1))))
            print(f"Ajout d'un client fictif avec demande {dummy_demand}")
        else:
            dummy_supply = total_demand - total_supply
            supply = np.append(supply, dummy_supply)
            costs = np.vstack((costs, np.zeros((1, costs.shape[1]))))
            print(f"Ajout d'une usine fictive avec capacité {dummy_supply}")

    # Méthode du coin Nord-Ouest
    num_sources = len(supply)
    num_destinations = len(demand)
    solution = np.zeros((num_sources, num_destinations))

    i, j = 0, 0
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()

    while i < num_sources and j < num_destinations:
        allocation = min(remaining_supply[i], remaining_demand[j])
        solution[i, j] = allocation
        remaining_supply[i] -= allocation
        remaining_demand[j] -= allocation

        if remaining_supply[i] == 0:
            i += 1
        if remaining_demand[j] == 0:
            j += 1

    print("\nSolution initiale (méthode du coin nord-ouest):")
    print(solution)

    # Calcul des coûts relatifs
    u = np.zeros(num_sources)
    v = np.zeros(num_destinations)
    basic_cells = [(i, j) for i in range(num_sources) for j in range(num_destinations) if solution[i, j] > 0]

    u[0] = 0  # On fixe u1 à 0 arbitrairement

    # Résolution du système pour u et v
    changed = True
    while changed:
        changed = False
        for i, j in basic_cells:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = costs[i, j] - u[i]
                changed = True
            elif np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = costs[i, j] - v[j]
                changed = True

    # Calcul des coûts relatifs
    relative_costs = np.zeros((num_sources, num_destinations))
    for i in range(num_sources):
        for j in range(num_destinations):
            if solution[i, j] == 0:  # Seulement pour les cellules non basiques
                relative_costs[i, j] = costs[i, j] - u[i] - v[j]

    print("\nCoûts relatifs pour les cellules non basiques:")
    print(relative_costs)

    # Vérification de l'optimalité
    if np.all(relative_costs >= 0):
        print("\nLa solution est optimale!")
    else:
        print("\nLa solution peut être améliorée.")

    # Calcul du coût total
    total_cost = np.sum(costs * solution)
    print(f"\nCoût total de la solution: {total_cost}")

    # Affichage sous forme de DataFrame
    solution_df = pd.DataFrame(
        solution,
        index=[f"Usine {chr(65 + i)}" for i in range(num_sources)],
        columns=[f"Client {j + 1}" for j in range(num_destinations)]
    )
    print("\nPlan de transport optimal:")
    print(solution_df)

    # Visualisation du réseau de transport
    plt.figure(figsize=(10, 6))
    G = nx.DiGraph()

    # Ajout des nœuds
    for i in range(num_sources):
        G.add_node(f"Usine {chr(65 + i)}", node_color='lightblue', node_size=2000)

    for j in range(num_destinations):
        G.add_node(f"Client {j + 1}", node_color='lightgreen', node_size=2000)

    # Ajout des arêtes avec les quantités transportées
    edge_labels = {}
    for i in range(num_sources):
        for j in range(num_destinations):
            if solution[i, j] > 0:
                G.add_edge(
                    f"Usine {chr(65 + i)}",
                    f"Client {j + 1}",
                    weight=solution[i, j],
                    label=f"{solution[i, j]} (coût: {costs[i, j]})"
                )
                edge_labels[(f"Usine {chr(65 + i)}", f"Client {j + 1}")] = f"{solution[i, j]}"

    # Positionnement des nœuds
    pos = {}
    for i in range(num_sources):
        pos[f"Usine {chr(65 + i)}"] = (0, num_sources - i)

    for j in range(num_destinations):
        pos[f"Client {j + 1}"] = (2, num_destinations - j)

    # Dessin du graphe
    node_colors = ['lightblue' if 'Usine' in node else 'lightgreen' for node in G.nodes()]
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=10,
        font_weight='bold',
        arrowsize=20
    )

    # Ajout des labels sur les arêtes
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=9
    )

    plt.title("Réseau de transport optimal", fontsize=14)
    plt.tight_layout()
    plt.savefig('transport_optimal.png', dpi=300)
    plt.show()

    return solution, total_cost


def solve_production_problem():
    print("\n=== PROBLÈME DE PRODUCTION ===")

    # Définition du problème
    c = np.array([3, 2])  # Coefficients de la fonction objectif (profit)
    A = np.array([
        [2, 1],  # Contrainte sur la machine M1
        [1, 3]  # Contrainte sur la machine M2
    ])
    b = np.array([100, 90])  # Capacités des machines
    constraint_types = ["<=", "<="]

    # Résolution avec notre solveur
    solver = SimplexSolver()
    solver.define_problem(c.tolist(), A.tolist(), b.tolist(), constraint_types, "max")
    result = solver.solve()

    # Affichage des résultats
    print("\nSolution optimale:")
    print(f"Quantité de P1 à produire: {result['variables'][0]:.2f} unités")
    print(f"Quantité de P2 à produire: {result['variables'][1]:.2f} unités")
    print(f"Profit maximal: {result['objective_value']:.2f} €")

    # Visualisation graphique
    plt.figure(figsize=(10, 6))

    # Tracé des contraintes
    x = np.linspace(0, 60, 400)
    plt.plot(x, (100 - 2 * x) / 1, label='2x1 + x2 ≤ 100 (M1)')
    plt.plot(x, (90 - x) / 3, label='x1 + 3x2 ≤ 90 (M2)')

    # Zone de solution réalisable
    y1 = np.minimum((100 - 2 * x) / 1, (90 - x) / 3)
    y1 = np.maximum(y1, 0)
    plt.fill_between(x, 0, y1, where=(y1 >= 0), color='lightgreen', alpha=0.3)

    # Solution optimale
    plt.scatter(result['variables'][0], result['variables'][1], color='red', s=100,
                label=f'Solution optimale ({result["variables"][0]:.1f}, {result["variables"][1]:.1f})')

    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.xlabel('Quantité de P1')
    plt.ylabel('Quantité de P2')
    plt.title('Espace des solutions - Problème de production')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('production_optimale.png', dpi=300)
    plt.show()

    return result


if __name__ == "__main__":
    # Résolution du problème de transport
    transport_solution, transport_cost = solve_transportation_problem()

    # Résolution du problème de production
    production_result = solve_production_problem()

    print("\n=== SYNTHÈSE DES RÉSULTATS ===")
    print(f"Coût total minimum du transport: {transport_cost:.2f} €")
    print(f"Profit maximum de la production: {production_result['objective_value']:.2f} €")