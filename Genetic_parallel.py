import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score


def split(data, label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(data, label, test_size=0.3, random_state=42)
    return X_tr, X_te, Y_tr, Y_te


def acc_scor(data, label):
    X_train, X_test, Y_train, Y_test = split(data_bc, label_bc)
    model = LogisticRegression().fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(Y_test, predictions, normalize=True)
    return acc


def plot(score, x, y, c="b"):
    gen = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
           20]  # ,21,22,23,24,25]#,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]  #
    plt.figure(figsize=(25, 4))
    ax = sns.pointplot(x=gen, y=score, color=c)
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x, y))
    plt.show()


def init(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool)
        chromosome[:int(0.25 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population, X_train, X_test, Y_train, Y_test):
    scores = []
    for individual in population:
        cols = [index for index in range(len(individual)) if individual[index] == True]
        model = LogisticRegression().fit(X_train.iloc[:, cols], Y_train)
        prediction = model.predict(X_test.iloc[:, cols])
        acc = accuracy_score(Y_test, prediction)
        a = 1
        scores.append(acc * a + (1 - a) * (1 - sum(individual) / len(individual)))

    return scores


def selection(pop_after_fit, scores, n_parents):
    pop = []
    scores_selected = []

    inds = np.argsort(scores)

    print(scores)

    print(inds)
    print(len(scores))

    for i in range(n_parents):
        pop.append(pop_after_fit[inds[len(inds) - 1 - i]])
        scores_selected.append(scores[inds[len(inds) - 1 - i]])
    print(scores_selected)
    return pop, scores_selected


def crossover(pop_after_sel, scores):
    pop = []
    inds = np.argsort(scores)
    pop.append(pop_after_sel[inds[-1]])
    sum_scores = sum(scores)
    prob_list = scores / sum_scores
    progenitor_list_a = np.random.choice(list(range(len(pop_after_sel))), len(pop_after_sel), p=prob_list, replace=True)
    progenitor_list_b = np.random.choice(list(range(len(pop_after_sel))), len(pop_after_sel), p=prob_list, replace=True)

    for i in range(len(pop_after_sel) - 1):
        new_par1 = []
        new_par2 = []
        child_1 = pop_after_sel[progenitor_list_a[i]]
        child_2 = pop_after_sel[progenitor_list_b[i]]
        new_par = np.concatenate((child_1[:len(child_1) // 2], child_2[len(child_1) // 2:]))
        pop.append(new_par)
    return pop


def mutation(pop_after_cross, mutation_rate, n_feat):
    pop = []
    mutation_range = int(mutation_rate * n_feat)
    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range):
            pos = randint(0, n_feat - 1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop.append(chromo)
    return pop


def evolve_slow(pop_after_cross, mutation_rate, n_feat):
    pop = []
    mutation_range = int(mutation_rate * n_feat)
    pop.append(pop_after_cross[0])
    for n in range(1, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range):
            pos = randint(0, n_feat - 1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop.append(chromo)
    return pop


def evolve_fast(pop_after_cross, mutation_rate, n_feat):
    pop = []
    mutation_range = int(mutation_rate * n_feat)
    pop.append(pop_after_cross[0])
    for n in range(1, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range * 5):
            pos = randint(0, n_feat - 1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop.append(chromo)
    return pop


def generations(df, label, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
    best_chromo = []
    best_score = []
    best_chromo_count = []
    population_nextgen = init(n_parents, n_feat)

    for i in range(n_gen):
        scores = fitness_score(population_nextgen, X_train, X_test, Y_train, Y_test)
        pop_after_sel, scores_after_selection = selection(population_nextgen, scores, n_parents)
        print('Best score in generation', i + 1, ':', scores_after_selection[0])  # 2
        print("pop num after selection =", len(pop_after_sel))
        pop_after_cross = crossover(pop_after_sel, scores_after_selection)
        #population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)
        population_slow = evolve_slow(pop_after_cross, mutation_rate, n_feat)
        population_fast = evolve_fast(pop_after_cross, mutation_rate, n_feat)
        population_nextgen = population_slow
        population_nextgen.extend(population_fast)
        best_chromo.append(pop_after_sel[0])
        best_score.append(scores_after_selection[0])
        print(sum(pop_after_sel[0]))
        best_chromo_count.append(sum(pop_after_sel[0]))
        print("best chromo =", best_chromo[i], "best score =", best_score[i])
    print("en iyi kromozomlar", best_chromo)
    print("en iyi skorlar", best_score)
    print("ortalama=", sum(best_score) / len(best_score))
    print("ortalama=", best_chromo_count)
    return best_chromo, best_score


if __name__ == '__main__':

    data_bc = pd.read_csv("/home/yunus/Desktop/PGA/tez/data.csv")
    label_bc = data_bc["diagnosis"]
    label_bc = np.where(label_bc == 'M', 1, 0)
    print(label_bc)
    data_bc.drop(["id", "diagnosis", "Unnamed: 32"], axis=1, inplace=True)
    rows = []
    for i in range(data_bc.shape[0]):
        rows.append(
            [np.random.normal(0, 0.5, 1), np.random.normal(0, 0.5, 1), np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1), np.random.normal(0, 0.5, 1), np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1), np.random.normal(0, 0.5, 1), np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1),
             np.random.normal(0, 0.5, 1)])
    df = pd.DataFrame(rows,
                      columns=["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14",
                               "N15"])
   # data_bc = pd.concat([data_bc, df], axis=1, join='inner')

    print("Breast Cancer dataset:\n", data_bc.shape[0], "Records\n", data_bc.shape[1], "Features")
    print(data_bc.head())
    print("All the features in this dataset have continuous values")
    X_train, X_test, Y_train, Y_test = split(data_bc, label_bc)
    score1 = acc_scor(data_bc, label_bc)
    print("LogisticRegression skoru =", score1)
    print("data shape 1 = ", data_bc.shape[1])

    chromo_df_bc, score_bc = generations(data_bc, label_bc, n_feat=data_bc.shape[1], n_parents=20,
                                         mutation_rate=0.1, n_gen=20, X_train=X_train, X_test=X_test, Y_train=Y_train,
                                         Y_test=Y_test)

    plot(score_bc, 0.8, 1.0, c="gold")
