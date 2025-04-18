import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap


class ProjetPERT:
    def __init__(self):
        self.taches = {}  # {nom_tache: {"duree": duree, "anteriorites": [liste]}}
        self.graphe = nx.DiGraph()
        self.dates_tot = {}  # Dates au plus tôt
        self.dates_tard = {}  # Dates au plus tard
        self.marges = {}  # Marges libres
        self.chemin_critique = []
        self.date_debut_projet = 0
        self.date_fin_projet = 0

    def ajouter_tache(self, nom, duree, anteriorites=None):
        """Ajoute une tâche au projet"""
        if anteriorites is None:
            anteriorites = []
        self.taches[nom] = {"duree": duree, "anteriorites": anteriorites}

    def charger_taches_depuis_csv(self, fichier_csv):
        """Version robuste qui gère les virgules dans les antériorités"""
        with open(fichier_csv, 'r') as f:
            lines = f.readlines()

        # Traitement manuel pour éviter les problèmes de parsing
        for line in lines[1:]:  # On ignore l'en-tête
            line = line.strip()
            if not line:
                continue

            # Séparation spéciale qui ne split que les 2 premières virgules
            parts = line.split(',', 2)
            if len(parts) < 3:
                parts += ['']  # Ajoute une valeur vide si antériorités manquantes

            nom, duree, anteriorites = parts
            anteriorites = [a.strip() for a in anteriorites.split(',') if a.strip()]

            try:
                self.ajouter_tache(nom.strip(), int(duree), anteriorites)
            except ValueError as e:
                print(f"Erreur ligne '{line}': {str(e)}")
    def charger_taches_depuis_dataframe(self, df):
        """Charge les tâches à partir d'un DataFrame pandas"""
        for _, row in df.iterrows():
            nom = row['tache']
            duree = row['duree']
            anteriorites = row['anteriorites'].split(',') if pd.notna(row['anteriorites']) else []
            self.ajouter_tache(nom, duree, anteriorites)

    def construire_graphe(self):
        """Construit le graphe PERT à partir des tâches"""
        # Créer les nœuds pour chaque tâche
        for nom_tache, info in self.taches.items():
            self.graphe.add_node(nom_tache, duree=info["duree"])

        # Ajouter les arcs pour les antériorités
        for nom_tache, info in self.taches.items():
            for ant in info["anteriorites"]:
                self.graphe.add_edge(ant, nom_tache)

        # Ajouter des nœuds "Début" et "Fin" virtuels si nécessaire
        taches_sans_predecesseur = [t for t in self.taches if not self.taches[t]["anteriorites"]]
        taches_sans_successeur = [t for t in self.taches if not list(self.graphe.successors(t))]

        if len(taches_sans_predecesseur) > 1:
            self.graphe.add_node("Début", duree=0)
            for t in taches_sans_predecesseur:
                self.graphe.add_edge("Début", t)

        if len(taches_sans_successeur) > 1:
            self.graphe.add_node("Fin", duree=0)
            for t in taches_sans_successeur:
                self.graphe.add_edge(t, "Fin")

    def calculer_dates_tot(self):
        """Calcule les dates au plus tôt pour chaque tâche"""
        self.dates_tot = {tache: 0 for tache in self.graphe.nodes}

        # Tri topologique pour traiter les tâches dans l'ordre
        for tache in nx.topological_sort(self.graphe):
            predecesseurs = list(self.graphe.predecessors(tache))
            if predecesseurs:
                # La date au plus tôt est le maximum des dates de fin des prédécesseurs
                self.dates_tot[tache] = max(self.dates_tot[pred] + self.graphe.nodes[pred]["duree"]
                                            for pred in predecesseurs)

    def calculer_dates_tard(self):
        """Calcule les dates au plus tard pour chaque tâche"""
        # Initialiser avec une grande valeur
        self.dates_tard = {tache: float('inf') for tache in self.graphe.nodes}

        # Trouver la date de fin du projet (date au plus tôt du dernier nœud)
        self.date_fin_projet = max(self.dates_tot[tache] + self.graphe.nodes[tache]["duree"]
                                   for tache in self.graphe.nodes)

        # Pour les tâches sans successeur, la date au plus tard = date fin projet - durée
        for tache in self.graphe.nodes:
            if not list(self.graphe.successors(tache)):
                self.dates_tard[tache] = self.date_fin_projet - self.graphe.nodes[tache]["duree"]

        # Parcourir le graphe dans l'ordre topologique inverse
        for tache in reversed(list(nx.topological_sort(self.graphe))):
            successeurs = list(self.graphe.successors(tache))
            if successeurs:
                # La date au plus tard est le minimum des dates de début au plus tard des successeurs
                self.dates_tard[tache] = min(self.dates_tard[succ] for succ in successeurs) - self.graphe.nodes[tache][
                    "duree"]

    def calculer_marges(self):
        """Calcule les marges pour chaque tâche"""
        for tache in self.graphe.nodes:
            self.marges[tache] = self.dates_tard[tache] - self.dates_tot[tache]

    def trouver_chemin_critique(self):
        """Identifie le chemin critique (tâches avec marge nulle)"""
        self.chemin_critique = [tache for tache in self.graphe.nodes
                                if self.marges.get(tache, float('inf')) == 0
                                and tache not in ["Début", "Fin"]]

    def analyser_projet(self):
        """Effectue l'analyse complète du projet"""
        self.construire_graphe()
        self.calculer_dates_tot()
        self.calculer_dates_tard()
        self.calculer_marges()
        self.trouver_chemin_critique()

    def afficher_resultats(self):
        """Affiche les résultats de l'analyse"""
        resultats = {
            "Tâche": [],
            "Durée": [],
            "Date au plus tôt": [],
            "Date au plus tard": [],
            "Marge": [],
            "Sur chemin critique": []
        }

        for tache in sorted(self.graphe.nodes):
            if tache not in ["Début", "Fin"]:
                resultats["Tâche"].append(tache)
                resultats["Durée"].append(self.graphe.nodes[tache]["duree"])
                resultats["Date au plus tôt"].append(self.dates_tot[tache])
                resultats["Date au plus tard"].append(self.dates_tard[tache])
                resultats["Marge"].append(self.marges[tache])
                resultats["Sur chemin critique"].append("Oui" if tache in self.chemin_critique else "Non")

        return pd.DataFrame(resultats)

    def dessiner_graphe(self, figsize=(12, 8), save_path=None):
        """Dessine le graphe PERT avec un layout hiérarchique amélioré"""
        plt.figure(figsize=figsize)

        # Utiliser un layout hiérarchique pour une meilleure organisation
        niveaux = {}
        for node in nx.topological_sort(self.graphe):
            if not list(self.graphe.predecessors(node)):
                niveaux[node] = 0
            else:
                niveaux[node] = 1 + max(niveaux[pred] for pred in self.graphe.predecessors(node))

        pos = {}
        niveaux_count = {}
        for niveau in set(niveaux.values()):
            niveaux_count[niveau] = 0

        for node in nx.topological_sort(self.graphe):
            niveau = niveaux[node]
            count = niveaux_count[niveau]
            niveaux_count[niveau] += 1

            nodes_in_level = sum(1 for n, l in niveaux.items() if l == niveau)
            x_spacing = 1.0 / (nodes_in_level + 1) if nodes_in_level > 0 else 0.5

            pos[node] = (niveau, 0.5 - count * x_spacing * 2)

        max_marge = max(self.marges.values()) if self.marges else 0
        cmap = LinearSegmentedColormap.from_list('marge_cmap', ['#FF4136', '#FFDC00', '#2ECC40'])

        node_colors = []
        for node in self.graphe.nodes:
            if node in self.chemin_critique:
                node_colors.append('#FF4136')
            else:
                normalized_marge = self.marges[node] / max_marge if max_marge > 0 else 0
                node_colors.append(cmap(normalized_marge))

        nx.draw_networkx_nodes(self.graphe, pos,
                               node_color=node_colors,
                               node_size=700,
                               alpha=0.8,
                               node_shape='o',
                               edgecolors='black',
                               linewidths=1.5)

        edge_colors = ['#FF4136' if u in self.chemin_critique and v in self.chemin_critique
                       else '#333333' for u, v in self.graphe.edges()]

        nx.draw_networkx_edges(self.graphe, pos,
                               edge_color=edge_colors,
                               width=2.0,
                               arrowsize=15,
                               alpha=0.8,
                               arrows=True)

        # Ajouter les étiquettes de toutes les tâches
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.1, f"{node} ({self.graphe.nodes[node]['duree']})", fontsize=10, ha='center', va='center',
                     fontweight='bold')
            plt.text(x, y - 0.1, f"TOT: {self.dates_tot[node]}", fontsize=8, ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.2'))
            plt.text(x, y - 0.2, f"TARD: {self.dates_tard[node]}", fontsize=8, ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.2'))

        legend_elements = [
            Patch(facecolor='#FF4136', edgecolor='black', label='Chemin critique'),
            Patch(facecolor='#FFDC00', edgecolor='black', label='Marge moyenne'),
            Patch(facecolor='#2ECC40', edgecolor='black', label='Marge élevée')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title("Graphe PERT - Analyse du projet", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generer_gantt(self, figsize=(14, 8), save_path=None):
        """Génère un diagramme de Gantt amélioré"""
        fig, ax = plt.subplots(figsize=figsize)

        # Trier les tâches par date de début
        sorted_tasks = sorted([(t, self.dates_tot[t]) for t in self.graphe.nodes
                               if t not in ["Début", "Fin"]], key=lambda x: (x[1], x[0]))

        # Créer le diagramme
        y_positions = range(len(sorted_tasks))
        task_names = [task[0] for task in sorted_tasks]

        # Palette de couleurs pour les marges
        max_marge = max(self.marges.values()) if self.marges else 0
        cmap = LinearSegmentedColormap.from_list('marge_cmap', ['#FF4136', '#FFDC00', '#2ECC40'])

        # Ajouter des grilles pour les dates importantes
        ax.grid(True, axis='x', linestyle='--', alpha=0.7, zorder=0)

        # Pour stocker les positions des barres pour ajouter les marges
        task_bars = []

        for i, (task, start) in enumerate(sorted_tasks):
            duration = self.graphe.nodes[task]["duree"]
            is_critical = task in self.chemin_critique
            marge = self.marges[task]

            if is_critical:
                color = '#FF4136'  # Rouge pour le chemin critique
            else:
                # Utiliser la palette de couleurs basée sur la marge
                normalized_marge = marge / max_marge if max_marge > 0 else 0
                color = cmap(normalized_marge)

            # Dessiner la barre de tâche
            bar = ax.barh(i, duration, left=start, color=color, alpha=0.8,
                          edgecolor='black', linewidth=1, zorder=3)
            task_bars.append((bar, task, start, duration, marge))

            # Ajouter l'étiquette sur la barre
            ax.text(start + duration / 2, i, f"{task} ({duration})",
                    ha='center', va='center', color='white', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.4, boxstyle="round,pad=0.1"))

            # Ajouter la marge comme une barre transparente si > 0
            if marge > 0:
                ax.barh(i, marge, left=start + duration, color='gray', alpha=0.3,
                        edgecolor='black', linestyle='dotted', linewidth=1, zorder=2)
                ax.text(start + duration + marge / 2, i, f"M:{marge}",
                        ha='center', va='center', color='black', fontsize=8)

        # Ajouter les dates au plus tôt et au plus tard
        for bar, task, start, duration, marge in task_bars:
            # Date au plus tôt (TOT)
            ax.text(start - 0.3, bar[0].get_y() + bar[0].get_height() / 2,
                    f"{start}", ha='right', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.1'))

            # Date au plus tard (TARD)
            tard = self.dates_tard[task] + self.graphe.nodes[task]["duree"]
            ax.text(start + duration + marge + 0.3, bar[0].get_y() + bar[0].get_height() / 2,
                    f"{tard}", ha='left', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.1'))

        # Paramètres du graphique
        ax.set_yticks(y_positions)
        ax.set_yticklabels(task_names)
        ax.set_xlabel('Temps', fontweight='bold')
        ax.set_title('Diagramme de Gantt du Projet', fontsize=14, fontweight='bold')

        # Étendre les limites pour avoir un peu d'espace
        x_max = self.date_fin_projet + 1
        ax.set_xlim(-1, x_max)

        # Lignes de grille horizontales
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.8)

        # Légende
        legend_elements = [
            Patch(facecolor='#FF4136', edgecolor='black', label='Chemin critique'),
            Patch(facecolor='#FFDC00', edgecolor='black', label='Marge moyenne'),
            Patch(facecolor='#2ECC40', edgecolor='black', label='Marge élevée'),
            Patch(facecolor='gray', alpha=0.3, edgecolor='black', linestyle='dotted', label='Marge')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.01))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def exporter_tableau_resultats(self, filepath="resultats_projet.xlsx"):
        """Exporte les résultats du projet dans un fichier Excel"""
        df = self.afficher_resultats()
        df.to_excel(filepath, index=False)
        print(f"Résultats exportés dans {filepath}")

    def exporter_visualisations(self, prefix="projet_pert"):
        """Exporte toutes les visualisations"""
        # Exporter le graphe PERT
        self.dessiner_graphe(save_path=f"{prefix}_graphe.png")

        # Exporter le diagramme de Gantt
        self.generer_gantt(save_path=f"{prefix}_gantt.png")

        # Exporter les résultats
        self.exporter_tableau_resultats(filepath=f"{prefix}_resultats.xlsx")

        print(f"Toutes les visualisations ont été exportées avec le préfixe '{prefix}'")


def main():
    if len(sys.argv) > 1:
        # Si un argument est passé (comme le nom du fichier CSV)
        fichier_csv = sys.argv[1]
        projet = ProjetPERT()
        try:
            projet.charger_taches_depuis_csv(fichier_csv)
            print(f"Tâches chargées depuis {fichier_csv}")
        except FileNotFoundError:
            print(f"Erreur: Fichier {fichier_csv} non trouvé")
            return
        except Exception as e:
            print(f"Erreur lors du chargement du CSV: {str(e)}")
            return
    else:
        # Sinon, utiliser des données par défaut
        print("Aucun fichier spécifié, utilisation des données par défaut...")
        taches = {
            'A': {'duree': 4, 'anteriorites': []},
            'B': {'duree': 8, 'anteriorites': []},
            'C': {'duree': 1, 'anteriorites': []},
            'D': {'duree': 1, 'anteriorites': ['C']},
            'E': {'duree': 6, 'anteriorites': ['A']},
            'F': {'duree': 3, 'anteriorites': ['A']},
            'G': {'duree': 5, 'anteriorites': ['B']},
            'H': {'duree': 3, 'anteriorites': ['E', 'F', 'G']},
            'I': {'duree': 1, 'anteriorites': ['D']},
            'J': {'duree': 2, 'anteriorites': ['I']},
            'K': {'duree': 2, 'anteriorites': ['H']},
            'L': {'duree': 5, 'anteriorites': ['J', 'K']}
        }
        projet = ProjetPERT()
        projet.charger_taches_depuis_dict(taches)

    # Vérification que des tâches ont bien été chargées
    if not projet.taches:
        print("Erreur: Aucune tâche n'a été chargée dans le projet")
        return

    try:
        projet.analyser_projet()

        print("\nRésultats de l'analyse:")
        print(projet.afficher_resultats())
        print("\nChemin critique:", projet.chemin_critique)
        print(f"Durée totale du projet: {projet.date_fin_projet} unités de temps")

        projet.dessiner_graphe()
        projet.generer_gantt()

        # Exporter les résultats (optionnel)
        # projet.exporter_visualisations(prefix="mon_projet")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'analyse: {str(e)}")


if __name__ == "__main__":
    main()