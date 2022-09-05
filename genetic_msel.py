# -----------------------------
# ---------- PARTE 1 ----------
# -----------------------------

# --------------
# --- IMPORT ---
# --------------

import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from itersel import *
from genetic.problem import Element, GAProblem, MultiThreadGAProblem
from genetic.evolution import GeneticSolver, MultiThreadGeneticSolver
from genetic.crossover import uniform_crossover
import matplotlib.pyplot as plt

# ---------------
# --- DATASET ---
# ---------------

# Caricamento
credit_df = pd.read_csv("dataset/credit.csv")

# Info sul dataset
credit_df.head()
credit_df.info()
credit_df.describe()

# Estrazione della variabile target
X_credit = credit_df.drop("Balance", axis = 1)
Y_credit = np.log1p(credit_df["Balance"])

# Encoding delle variabili categoriche
X_credit = pd.get_dummies(X_credit, drop_first=True)
X_credit.head()

# --------------------------
# --- ALGORITMO GENETICO ---
# --------------------------

# Funzione obiettivo
def eval_lin_reg(X: pd.DataFrame, Y: pd.Series, return_model: bool = False):
    '''
        Restituisce il valore del criterio di valutazione selezionato di un modello 
        di Regressione Lineare costruito sulla base dei dati X ed Y.

        Parametri:

        - `X`: Variabili di input del modello di Regressione Lineare
        - `Y`: Variabili di output del modello di Regressione Lineare
        - `return_model`: indica se il modello costruito deve essere restituito dalla funzione. Default = `False`

        Return:

        La funzione restituisce un oggetto contenente i valori dei criteri di selezione, 
        il Mean Squared Error, il Residual Sum of Squares ed il modello costruito (se `return_model = True`).
    '''
    # Definizione del modello
    model = LinearRegression()
    # Addestramento
    model.fit(X, Y)
    # Predizione
    Y_pred = model.predict(X)
    # Mean Squared Error
    mse = mean_squared_error(Y, Y_pred)
    # Residual Sum of Squares
    rss = np.sum((Y - Y_pred) ** 2)
    # Numero di parametri del modello
    num_params = len(model.coef_) + 1
    # Numero di record del dataset
    n = len(Y)
    # AIC
    aic = n * np.log(mse) + 2 * num_params
    # BIC
    bic = n * np.log(mse) + np.log(n) * num_params
    # Adj. R^2
    Y_bar = np.full(shape = (n,), fill_value = np.sum(Y, axis = 0) / n)
    tss = np.sum((Y - Y_bar) ** 2, axis = 0)
    rss = np.sum((Y - Y_pred) ** 2, axis = 0)
    adjr = 1 - (rss / (n - num_params - 1)) / (tss / (n - 1))
    
    result = {
        AIC:aic, BIC:bic, ADJ_R2:adjr, RSS:rss, "_coef": model.coef_
    }
    if return_model:
        result["model"] = model
    return result


# Definizione del Problema
#
#  Caratteristiche:
#     - Soluzione = vettore di valori booleani di dimensione pari al numero di features del dataset. 
#                   Ogni valore indica se la corrispondente variabile deve essere utilizzata o meno nella costruzione del modello
#     - Crossover = Crossover Uniforme
#     - Mutazione = Negazione di un elemento casuale
#     - Meccanismo di correzione = Qualora una soluzione dovesse presentare tutti valori negativi (False), viene reso positivo un elemento a caso
#
class ModelSelectionProblem(GAProblem):
    '''
        Definizione di un problema di Model Selection risolvibile tramite Algoritmi Genetici
    '''
    def __init__(self, X: pd.DataFrame, Y: pd.Series, colnames: np.ndarray = None, criterion: int = AIC, transform_categorical: bool = False) -> None:
        self.X = X
        self.Y = Y
        if colnames == None:
            self.colnames = self.X.columns.to_numpy()
        else:
            self.colnames = colnames
        self.sol_dim = len(self.colnames)
        if criterion != AIC and criterion != BIC and criterion != ADJ_R2:
            raise ValueError("Invalid selection criterion. Must be AIC or BIC or ADJ_R2.")
        self.criterion = criterion
        self.transform_categorical = transform_categorical        

    def correct_solution(self, sol: np.ndarray):
        '''
            Funzione di correzione delle soluzioni.

            Se tutti gli elementi della soluzione sono `False`, 
            viene scelto casualmente un elemento ed impostato a `True`
        '''
        if np.sum(sol) == 0:
            sol[random.randrange(0, self.sol_dim)] = True

    def init_pop(self, n: int) -> list:
        pop = []
        for _ in range(n):
            elm = np.random.choice([False, True], self.sol_dim, replace = True)
            # Correzione
            self.correct_solution(elm)
            pop.append(elm)
        return pop

    def evaluate(self, elm: Element) -> None:
        sol = elm.solution
        # Costruzione del dataset a partire dalla soluzione
        cols = self.colnames[sol]
        X_eval = self.X[cols]
        if self.transform_categorical:
            X_eval = pd.get_dummies(X_eval)
        # Valutazione del modello
        result = eval_lin_reg(X_eval, self.Y, return_model = False)
        elm.fitness = result[self.criterion]

    def crossover(self, sol1: object, sol2: object) -> 'list | tuple':
        _sol1 = self.copy(sol1)
        _sol2 = self.copy(sol2)
        # Crossover uniforme
        ch1, ch2, _ = uniform_crossover(_sol1, _sol2)
        ch1 = np.array(ch1, dtype=bool)
        ch2 = np.array(ch2, dtype=bool)
        # Correzione
        self.correct_solution(ch1)
        self.correct_solution(ch2)

        return ch1, ch2

    def mutate(self, sol: object) -> object:
        _sol = self.copy(sol)
        # Indice casuale
        idx = random.randrange(0, self.sol_dim)
        # Negazione
        _sol[idx] = not _sol[idx]
        # Correzione
        self.correct_solution(_sol)
        return _sol

    def copy(self, sol: object) -> object:
        return np.copy(sol)

problem = ModelSelectionProblem(X = X_credit, 
                                Y = Y_credit,
                                criterion = AIC,
                                transform_categorical = False)

# Algoritmo risolutivo
#
#  Caratteristiche:
#    - Dimensione popolazione iniziale = 50
#    - Epoche = 30
#    - Strategia di aggiornamento della popolazione = Aggiornamento Generazionale
#    - Strategia di selezione del mating pool = Roulette Wheel
#    - Elitismo = Sì
#    - Obiettivo = Minimizzazione
pop_size = 50
epochs = 30
solver = GeneticSolver(problem, 
                       pop_size = pop_size, 
                       p_cross = 0.9, 
                       p_mut = 0.3,
                       pop_update_strategy="generational", 
                       selection_strategy="roulette_wheel",
                       use_elit = True,
                       minimize = True)
#
# >>> Funzioni di scheduling delle probailità di mutazione e di crossover
#
def pmut_sched(p_mut, epoch, best, last_update, fit_history):
    '''
        Scheduler della probabilità di mutazione
    '''
    if last_update == 0:
        return 0.1
    return min(0.9, p_mut + 0.05)

def pcross_sched(p_cross, epoch, best, last_update, fit_history):
    '''
        Scheduler della probabilità di crossover
    '''
    if last_update == 0:
        return 0.9
    return max(0.1, p_cross - 0.05)
#
# >>> Soluzione
#
solver.use_p_mut_scheduler(pmut_sched)
solver.use_p_cross_scheduler(pcross_sched)
best_gaAIC = solver.run(epochs = epochs, verbose = False, seed = 123)
best_comb = X_credit.columns[best_gaAIC.solution]
print("Combinazione migliore:", best_comb.to_list())
print("Valore criterio:", best_gaAIC.fitness)
#
# >>> Cronologia Fitness migliore
#
print("Last Update:", epochs - solver.last_update) # Epoca dell'ultimo aggiornamento del miglior individuo 
plt.figure(figsize = (6,3))
plt.plot(solver.fit_history, "o-")
plt.title("Fitness History")
plt.xlabel("Epoch")
plt.ylabel("Fitness")
plt.show()

# -------------------------------------
# --- METODI DI SELEZIONE ITERATIVA ---
# -------------------------------------

# EXHAUSTIVE SCELCTION

# AIC
best_eAIC, n_tested = exhaustive_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", best_eAIC[COMB].to_list())
print("Valore criterio:", best_eAIC[CRT_VALUE])
print("Valore RSS:", best_eAIC[RSS])
# BIC
best_eBIC, n_tested = exhaustive_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", best_eBIC[COMB].to_list())
print("Valore criterio:", best_eBIC[CRT_VALUE])
print("Valore RSS:", best_eBIC[RSS])
# ADJ. R2
best_eR2, n_tested = exhaustive_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", best_eR2[COMB].to_list())
print("Valore criterio:", best_eR2[CRT_VALUE])
print("Valore RSS:", best_eR2[RSS])

# FORWARD SELECTION

# AIC
best_fAIC, n_tested = forward_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_fAIC[COMB] > 0].to_list())
print("Valore criterio:", best_fAIC[CRT_VALUE])
print("Valore RSS:", best_fAIC[RSS])
# BIC
best_fBIC, n_tested = forward_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_fBIC[COMB] > 0].to_list())
print("Valore criterio:", best_fBIC[CRT_VALUE])
print("Valore RSS:", best_fBIC[RSS])
# Adj. R2
best_fR2, n_tested = forward_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_fR2[COMB] > 0].to_list())
print("Valore criterio:", best_fR2[CRT_VALUE])
print("Valore RSS:", best_fR2[RSS])

# BACKWARD SELECTION

# AIC
best_bAIC, n_tested = backward_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_bAIC[COMB] > 0].to_list())
print("Valore criterio:", best_bAIC[CRT_VALUE])
print("Valore RSS:", best_bAIC[RSS])
# BIC
best_bBIC, n_tested = backward_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_bBIC[COMB] > 0].to_list())
print("Valore criterio:", best_bBIC[CRT_VALUE])
print("Valore RSS:", best_bBIC[RSS])
# Adj. R2
best_bR2, n_tested = backward_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_bR2[COMB] > 0].to_list())
print("Valore criterio:", best_bR2[CRT_VALUE])
print("Valore RSS:", best_bR2[RSS])

# STEPWISE SELECTION 

# AIC
best_sAIC, n_tested = stepwise_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_sAIC[COMB] > 0].to_list())
print("Valore criterio:", best_sAIC[CRT_VALUE])
print("Valore RSS:", best_sAIC[RSS])
# BIC
best_sBIC, n_tested = stepwise_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_sBIC[COMB] > 0].to_list())
print("Valore criterio:", best_sBIC[CRT_VALUE])
print("Valore RSS:", best_sBIC[RSS])
# Adj. R2
best_sR2, n_tested = stepwise_selection(X_credit, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)
print("Numero di modelli testati:", n_tested)
print("Combinazione migliore:", X_credit.columns[best_sR2[COMB] > 0].to_list())
print("Valore criterio:", best_sR2[CRT_VALUE])
print("Valore RSS:", best_sR2[RSS])

# -----------------
# --- CONFRONTO ---
# -----------------
#
# >>> Algoritmo genetico per indici BIC ed Adjusted R2
#
solver.problem.criterion = BIC
solver.minimize = True
best_gaBIC = solver.run(epochs = epochs, verbose = False, seed = 123)
BIC_fitH = solver.fit_history
BIC_lastUpdate = epochs - solver.last_update # Epoca dell'ultimo aggiornamento del miglior individuo 

solver.problem.criterion = ADJ_R2
solver.minimize = False
best_gaR2 = solver.run(epochs = epochs, verbose = False, seed = 123)
R2_fitH = solver.fit_history
R2_lastUpdate = epochs - solver.last_update

print("BIC Update:", best_gaBIC.fitness)
print("R2 Update:", best_gaR2.fitness)
print("Last BIC Update:", BIC_lastUpdate)
print("Last R2 Update:", R2_lastUpdate)

plt.figure(figsize = (10,3))
plt.subplot(1, 2, 1)
plt.plot(BIC_fitH, "o-")
plt.title("Fitness History (BIC)")
plt.xlabel("Epoch")
plt.ylabel("Fitness")

plt.subplot(1, 2, 2)
plt.plot(R2_fitH, "o-")
plt.title("Fitness History (Adj. R^2)")
plt.xlabel("Epoch")

plt.show()
#
# >>> Risultati
#
res_df = pd.DataFrame({"Exhaustive": [best_eAIC[CRT_VALUE], best_eBIC[CRT_VALUE], best_eR2[CRT_VALUE]],
                        "Forward": [best_fAIC[CRT_VALUE], best_fBIC[CRT_VALUE], best_fR2[CRT_VALUE]],
                        "Backward": [best_bAIC[CRT_VALUE], best_bBIC[CRT_VALUE], best_bR2[CRT_VALUE]],
                        "Stepwise": [best_sAIC[CRT_VALUE], best_sBIC[CRT_VALUE], best_sR2[CRT_VALUE]],
                        "GA":[best_gaAIC.fitness, best_gaBIC.fitness, best_gaR2.fitness]}, index = ["AIC", "BIC", "ADJ. R2"])
print(res_df)


# -----------------------------
# ---------- PARTE 2 ----------
# -----------------------------

# ---------------
# --- DATASET ---
# ---------------

car_price = pd.read_csv("dataset/car_price_clean.csv")

X_car = car_price.drop(["MSRP"], axis = 1)
Y_car = np.log1p(car_price["MSRP"])

X_car = pd.get_dummies(X_car, drop_first=True)
X_car.head()

# ---------------------------
# --- SELEZIONE ITERATIVA ---
# ---------------------------

# AIC
best_fAIC, _ = forward_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
best_bAIC, _ = backward_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
best_sAIC, _ = stepwise_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
# BIC
best_fBIC, _ = forward_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
best_bBIC, _ = backward_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
best_sBIC, _ = stepwise_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = BIC, minimize=True)
# Adj. R2
best_fR2, _ = forward_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)
best_bR2, _ = backward_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)
best_sR2, _ = stepwise_selection(X_car, Y_car, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = ADJ_R2, minimize=False)


# ------------------------------------
# --- ALGORITMO GENETICO PARALLELO ---
# ------------------------------------

class ParallelModelSelectionProblem(MultiThreadGAProblem):
    '''
        Definizione di un problema di Model Selection risolvibile tramite Algoritmi Genetici Paralleli
    '''
    def __init__(self, X: pd.DataFrame, Y: pd.Series, colnames: np.ndarray = None, criterion: int = AIC, transform_categorical: bool = False, p_True:float = 0.5) -> None:
        self.X = X
        self.Y = Y
        if colnames == None:
            self.colnames = self.X.columns.to_numpy()
        else:
            self.colnames = colnames
        self.sol_dim = len(self.colnames)
        if criterion != AIC and criterion != BIC and criterion != ADJ_R2:
            raise ValueError("Invalid selection criterion. Must be AIC or BIC or ADJ_R2.")
        self.criterion = criterion
        self.transform_categorical = transform_categorical
        self.p_True = p_True

    def correct_solution(self, sol: np.ndarray):
        '''
            Funzione di correzione delle soluzioni.

            Se tutti gli elementi della soluzione sono `False`, 
            viene scelto casualmente un elemento ed impostato a `True`
        '''
        if np.sum(sol) == 0:
            sol[random.randrange(0, self.sol_dim)] = True

    def init_pop(self, n: int) -> list:
        pop = []
        for _ in range(n):
            elm = np.random.choice([False, True], self.sol_dim, replace = True, p = [1-self.p_True, self.p_True])
            # Correzione
            self.correct_solution(elm)
            pop.append(elm)
        return pop

    def evaluate(self, elm: Element, thread_idx: int = 0) -> None:
        sol = elm.solution
        # Costruzione del dataset a partire dalla soluzione
        cols = self.colnames[sol]
        X_eval = self.X[cols]
        if self.transform_categorical:
            X_eval = pd.get_dummies(X_eval)
        # Valutazione del modello
        result = eval_lin_reg(X_eval, self.Y, return_model = False)
        elm.fitness = result[self.criterion]

    def crossover(self, sol1: object, sol2: object, thread_idx: int = 0) -> 'list | tuple':
        _sol1 = self.copy(sol1)
        _sol2 = self.copy(sol2)
        # Crossover uniforme
        ch1, ch2, _ = uniform_crossover(_sol1, _sol2)
        ch1 = np.array(ch1, dtype=bool)
        ch2 = np.array(ch2, dtype=bool)
        # Correzione
        self.correct_solution(ch1)
        self.correct_solution(ch2)

        return ch1, ch2

    def mutate(self, sol: object, thread_idx: int = 0) -> object:
        _sol = self.copy(sol)
        # Indice casuale
        idx = random.randrange(0, self.sol_dim)
        # Negazione
        _sol[idx] = not _sol[idx]
        # Correzione
        self.correct_solution(_sol)
        return _sol

    def copy(self, sol: object, thread_idx: int = 0) -> object:
        return np.copy(sol)

problem = ParallelModelSelectionProblem(X = X_car, 
                                        Y = Y_car,
                                        criterion = AIC)
#
# >>> Risoluzione parallela
#
pop_size = 150
epochs = 30
solver = MultiThreadGeneticSolver(problem, 
                                    pop_size = pop_size, 
                                    p_cross = 0.9, 
                                    p_mut = 0.3,
                                    pop_update_strategy="best_n", 
                                    selection_strategy="roulette_wheel",
                                    use_elit = True,
                                    minimize = True,
                                    n_jobs = 6)
solver.use_p_mut_scheduler(pmut_sched)
solver.use_p_cross_scheduler(pcross_sched)

print("AIC")
best_gaAIC = solver.run(epochs = epochs, verbose = False, seed = 123)
AIC_fitH = solver.fit_history
AIC_lastUpdate = epochs - solver.last_update

print("BIC")
solver.problem.criterion = BIC
solver.minimize = True
best_gaBIC = solver.run(epochs = epochs, verbose = False, seed = 123)
BIC_fitH = solver.fit_history
BIC_lastUpdate = epochs - solver.last_update

print("Adjusted R^2")
solver.problem.criterion = ADJ_R2
solver.minimize = False
best_gaR2 = solver.run(epochs = epochs, verbose = False, seed = 123)
R2_fitH = solver.fit_history
R2_lastUpdate = epochs - solver.last_update

print("Done")
#
# >>> Visualizzazione
#
plt.figure(figsize = (12,3))
plt.subplot(1, 3, 1)
plt.plot(AIC_fitH, "o-")
plt.title("Fitness History (AIC)")
plt.xlabel("Epoch")
plt.ylabel("Fitness")

plt.subplot(1, 3, 2)
plt.plot(BIC_fitH, "o-")
plt.title("Fitness History (BIC)")
plt.xlabel("Epoch")

plt.subplot(1, 3, 3)
plt.plot(R2_fitH, "o-")
plt.title("Fitness History (Adj. R^2)")
plt.xlabel("Epoch")

plt.show()
#
# >>> Confronto
#
res_df = pd.DataFrame({"Forward": [best_fAIC[CRT_VALUE], best_fBIC[CRT_VALUE], best_fR2[CRT_VALUE]],
                        "Backward": [best_bAIC[CRT_VALUE], best_bBIC[CRT_VALUE], best_bR2[CRT_VALUE]],
                        "Stepwise": [best_sAIC[CRT_VALUE], best_sBIC[CRT_VALUE], best_sR2[CRT_VALUE]],
                        "GA":[best_gaAIC.fitness, best_gaBIC.fitness, best_gaR2.fitness]}, index = ["AIC", "BIC", "ADJ. R2"])
res_df

# ------------------------------------
# --- COMBINAZIONE DEGLI ALGORITMI ---
# ------------------------------------
#
# >>> Problema modificato
#
class ModifiedModelSelectionProblem(GAProblem):
    '''
        Definizione di un problema di Model Selection risolvibile tramite Algoritmi Genetici
    '''
    def __init__(self, X: pd.DataFrame, Y: pd.Series, colnames: np.ndarray = None, criterion: int = AIC, transform_categorical: bool = False,
                       inject: np.ndarray = None, p_True:float = 0.5) -> None:
        self.X = X
        self.Y = Y
        if colnames == None:
            self.colnames = self.X.columns.to_numpy()
        else:
            self.colnames = colnames
        self.sol_dim = len(self.colnames)
        if criterion != AIC and criterion != BIC and criterion != ADJ_R2:
            raise ValueError("Invalid selection criterion. Must be AIC or BIC or ADJ_R2.")
        self.criterion = criterion
        self.transform_categorical = transform_categorical
        self.inject = inject   
        self.p_True = p_True

    def correct_solution(self, sol: np.ndarray):
        '''
            Funzione di correzione delle soluzioni.

            Se tutti gli elementi della soluzione sono `False`, 
            viene scelto casualmente un elemento ed impostato a `True`
        '''
        if np.sum(sol) == 0:
            sol[random.randrange(0, self.sol_dim)] = True

    def init_pop(self, n: int) -> list:
        if isinstance(self.inject, np.ndarray) and self.inject.any():
            pop = [self.inject]
        else:
            pop = []
        for _ in range(n):
            elm = np.random.choice([False, True], self.sol_dim, replace = True, p = [1-self.p_True, self.p_True])
            # Correzione
            self.correct_solution(elm)
            pop.append(elm)
        return pop

    def evaluate(self, elm: Element) -> None:
        sol = elm.solution
        # Costruzione del dataset a partire dalla soluzione
        cols = self.colnames[sol]
        X_eval = self.X[cols]
        if self.transform_categorical:
            X_eval = pd.get_dummies(X_eval)
        # Valutazione del modello
        result = eval_lin_reg(X_eval, self.Y, return_model = False)
        elm.fitness = result[self.criterion]

    def crossover(self, sol1: object, sol2: object) -> 'list | tuple':
        _sol1 = self.copy(sol1)
        _sol2 = self.copy(sol2)
        # Crossover uniforme
        ch1, ch2, _ = uniform_crossover(_sol1, _sol2)
        ch1 = np.array(ch1, dtype=bool)
        ch2 = np.array(ch2, dtype=bool)
        # Correzione
        self.correct_solution(ch1)
        self.correct_solution(ch2)

        return ch1, ch2

    def mutate(self, sol: object) -> object:
        _sol = self.copy(sol)
        # Indice casuale
        idx = random.randrange(0, self.sol_dim)
        # Negazione
        _sol[idx] = not _sol[idx]
        # Correzione
        self.correct_solution(_sol)
        return _sol

    def copy(self, sol: object) -> object:
        return np.copy(sol)
#
# >>> Modifica al dataset: aggiunta termini di interazione
#
polyFeat = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_credit_poly = polyFeat.fit_transform(X_credit)
X_credit_poly = pd.DataFrame(X_credit_poly)
X_credit_poly.columns = polyFeat.get_feature_names_out()
X_credit_poly.head()
#
# >>> FORWARD SELECTION con indice AIC
#
best_fAIC, _ = forward_selection(X_credit_poly, Y_credit, eval = lambda x, y : eval_lin_reg(x, y, False), criterion = AIC, minimize=True)
print(best_fAIC[CRT_VALUE])
#
# >>> Combinazione con ALGORITMO GENETICO
#
epochs = 40
problem = ModifiedModelSelectionProblem(X = X_credit_poly, 
                                Y = Y_credit,
                                criterion = AIC,
                                inject=best_fAIC["comb"].astype(bool))
solver = GeneticSolver(problem, 
                        pop_size = 50, 
                        p_cross = 0.9, 
                        p_mut = 0.1,
                        pop_update_strategy="best_n",
                        minimize = True)
solver.use_p_mut_scheduler(pmut_sched)
solver.use_p_cross_scheduler(pcross_sched)

best_gaAIC = solver.run(epochs = epochs, verbose = False, seed = 123)
AIC_fitH = solver.fit_history
AIC_lastUpdate = epochs - solver.last_update
print("AIC =", best_gaAIC.fitness)

plt.figure(figsize = (6,3))
plt.plot(solver.fit_history, "o-")
plt.title("Fitness History")
plt.xlabel("Epoch")
plt.ylabel("Fitness")
plt.show()