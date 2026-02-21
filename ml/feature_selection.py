import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from pyswarms.discrete import BinaryPSO
import random


# ==========================================
# 🔹 Fitness Function (Common for All)
# ==========================================
def fitness_function(X, y, mask):
    if np.count_nonzero(mask) == 0:
        return 1.0

    X_selected = X[:, mask == 1]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    score = cross_val_score(clf, X_selected, y, cv=3).mean()
    return 1 - score   # minimize


# ==========================================
# 🔹 PSO Feature Selection
# ==========================================
def fs_pso(X, y, feature_names):

    def objective(particles):
        n_particles = particles.shape[0]
        scores = np.zeros(n_particles)

        for i, particle in enumerate(particles):
            scores[i] = fitness_function(X, y, particle)

        return scores

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 1, 'p': 2}

    optimizer = BinaryPSO(
        n_particles=20,
        dimensions=X.shape[1],
        options=options
    )

    best_cost, best_position = optimizer.optimize(objective, iters=20)

    selected_idx = np.where(best_position == 1)[0]
    selected_features = [feature_names[i] for i in selected_idx]

    return X[:, selected_idx], selected_features


# ==========================================
# 🔹 GA Feature Selection
# ==========================================
def fs_ga(X, y, feature_names):

    population_size = 20
    generations = 20
    n_features = X.shape[1]

    population = np.random.randint(0, 2, (population_size, n_features))

    for _ in range(generations):

        fitness_scores = np.array([fitness_function(X, y, ind) for ind in population])
        sorted_idx = np.argsort(fitness_scores)
        population = population[sorted_idx]

        new_population = population[:5]  # elitism

        while len(new_population) < population_size:
            p1, p2 = random.sample(list(population[:10]), 2)
            crossover_point = random.randint(1, n_features - 1)

            child = np.concatenate((p1[:crossover_point], p2[crossover_point:]))

            # mutation
            if random.random() < 0.2:
                mutation_point = random.randint(0, n_features - 1)
                child[mutation_point] = 1 - child[mutation_point]

            new_population = np.vstack([new_population, child])

        population = new_population

    best = population[0]
    selected_idx = np.where(best == 1)[0]
    selected_features = [feature_names[i] for i in selected_idx]

    return X[:, selected_idx], selected_features


# ==========================================
# 🔹 ACO Feature Selection
# ==========================================
def fs_aco(X, y, feature_names):

    n_features = X.shape[1]
    pheromone = np.ones(n_features)

    best_mask = None
    best_score = 1.0

    for _ in range(20):

        mask = np.zeros(n_features)

        for i in range(n_features):
            if random.random() < pheromone[i] / pheromone.sum():
                mask[i] = 1

        score = fitness_function(X, y, mask)

        if score < best_score:
            best_score = score
            best_mask = mask

        pheromone = 0.9 * pheromone
        pheromone += (1 - score)

    selected_idx = np.where(best_mask == 1)[0]
    selected_features = [feature_names[i] for i in selected_idx]

    return X[:, selected_idx], selected_features


# ==========================================
# 🔥 HYBRID (PSO + GA + ACO)
# HYBRID WILL DOMINATE
# ==========================================
def fs_hybrid_advanced(X, y, feature_names):

    X_pso, f_pso = fs_pso(X, y, feature_names)
    X_ga, f_ga = fs_ga(X, y, feature_names)
    X_aco, f_aco = fs_aco(X, y, feature_names)

    # Combine best features from all
    combined_features = list(set(f_pso + f_ga + f_aco))

    selected_idx = [feature_names.index(f) for f in combined_features]

    return X[:, selected_idx], combined_features
