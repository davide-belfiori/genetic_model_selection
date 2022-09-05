'''
    Funzioni iterative di Model Selection
'''

# --------------
# --- IMPORT ---
# --------------

import itertools
from typing import Callable
import pandas as pd
import numpy as np

# ---------------
# --- GLOBALI ---
# ---------------

AIC = "aic"
BIC = "bic"
ADJ_R2 = "adjr"
RSS = "rss"
COMB = "comb"
CRT_VALUE ="crt_value"

# -----------------
# --- FUNCTIONS ---
# -----------------

def exhaustive_selection(X: pd.DataFrame, Y: pd.Series, 
                            eval: Callable[[pd.DataFrame, pd.Series], dict], 
                            criterion: str = AIC, 
                            minimize : bool = True):
    '''
        Model Selection Esaustivo.

        Testa tutte le combinazioni di variabili di `X` e restituisce la migliore.
    '''
    maximize = not minimize
    # Miglior risultato
    best = {COMB:None,
            CRT_VALUE:None,
            RSS:None}
    # Numero totale di variabili
    p = X.shape[1]
    # Numero di modelli testati
    n_tested = 0
    # Per ogni combinazione di variabili ...
    for comb in itertools.product([True, False], repeat = p):
        # ricavo il sotto-dataset e ...
        cols = X.columns[pd.Index(comb)]
        # ad eccezione del modello vuoto ...
        if len(cols) == 0:
            continue
        X_eval = X[cols]
        # valuto il modello
        result = eval(X_eval, Y)
        # Aggiorno il migliore
        if best[CRT_VALUE] == None or (minimize and result[criterion] < best[CRT_VALUE]) or (maximize and result[criterion] > best[CRT_VALUE]):
            best[COMB] = X.columns[pd.Index(comb)]
            best[CRT_VALUE] = result[criterion]
            best[RSS] = result[RSS]
        n_tested += 1
    return best, n_tested  


def build_dataset(deg_list, data: pd.DataFrame):
        '''
            Funzione di costruzione di un dataset di valutazione 
            a partire da una lista di interi che specifica il grado del polinomio associato ad 
            ogni variabile del dataset.
        '''
        X_built = pd.DataFrame()
        for i, deg in enumerate(deg_list):
            if deg > 0:
                # nome della colonna
                col = data.columns[i]
                # termini del polinomio
                for j in range(1,deg+1):
                    coln = col + "_" + str(j)
                    X_built[coln] = data[col].copy() ** j
        return X_built

def forward_selection(X: pd.DataFrame, Y: pd.Series, 
                        eval : Callable[[pd.DataFrame, pd.Series], dict], 
                        criterion: str, 
                        n: int = 1, 
                        minimize: bool = True):
    maximize = not minimize
    # Miglior risultato
    best = {COMB:None,
            CRT_VALUE:None,
            RSS:None}
    # Numero di variabili
    n_var = X.columns.size
    # Numero di modelli testati
    n_tested = 0
    # Combinazione corrente
    comb = np.zeros(shape = (n_var, ), dtype=int)

    while np.count_nonzero(comb) != n_var :
        # ... resetto il miglior risultato dell'iterazione
        best_iter = {COMB:None,
                     CRT_VALUE:None,
                     RSS:None}
        for i in range(n_var):
            if comb[i] > 0:
                continue
            # ... resetto il miglior risultato del polinomio
            best_deg = {COMB:None,
                        CRT_VALUE:None,
                        RSS:None}
            eval_comb = comb.copy()
            # ... per ogni grado
            for _ in range(n):
                # aumento il grado della variabile corrente
                eval_comb[i] += 1
                # costruisco il dataset con la combinazione corrente ...
                X_eval = build_dataset(eval_comb, X)
                # valuto il modello
                result = eval(X_eval, Y)
                # Aggiorno il migliore del polinomio
                if best_deg[CRT_VALUE] == None or (minimize and result[criterion] < best_deg[CRT_VALUE]) or (maximize and result[criterion] > best_deg[CRT_VALUE]):
                    best_deg[COMB] = eval_comb.copy()
                    best_deg[CRT_VALUE] = result[criterion]
                    best_deg[RSS] = result[RSS]
                n_tested += 1
            # Aggiorno il migliore dell'iterazione
            if best_iter[RSS] == None or best_deg[RSS] < best_iter[RSS]:
                best_iter[COMB] = best_deg[COMB]
                best_iter[CRT_VALUE] = best_deg[CRT_VALUE]
                best_iter[RSS] = best_deg[RSS]
        # Aggiorno la combinazione corrente
        comb = best_iter[COMB]
        # Aggiorno il migliore globale
        if best[CRT_VALUE] == None or (minimize and best_iter[CRT_VALUE] < best[CRT_VALUE]) or (maximize and best_iter[CRT_VALUE] > best[CRT_VALUE]):
            best[COMB] = best_iter[COMB]
            best[CRT_VALUE] = best_iter[CRT_VALUE]
            best[RSS] = best_iter[RSS]
    return best, n_tested

def backward_selection(X: pd.DataFrame, Y: pd.Series, 
                        eval : Callable[[pd.DataFrame, pd.Series], dict], 
                        criterion: str, 
                        n: int = 1, 
                        minimize: bool = True):
    maximize = not minimize
    # Numero di variabili
    n_var = X.columns.size
    # Combinazione corrente
    comb = np.full(fill_value=n, shape = (n_var, ), dtype=int)
    # costruisco il dataset con la combinazione corrente ...
    X_eval = build_dataset(comb, X)
    # valuto il modello
    result = eval(X_eval, Y)
    # Miglior risultato
    best = {COMB:comb,
            CRT_VALUE:result[criterion],
            RSS:result[RSS]}
    # Numero di modelli testati
    n_tested = 1
    while np.count_nonzero(comb) > 1 :
        # ... resetto il miglior risultato dell'iterazione
        best_iter = {COMB:None,
                     CRT_VALUE:None,
                     RSS:None}
        for i in range(n_var):
            if comb[i] == 0:
                continue
            # ... resetto il miglior risultato del polinomio
            best_deg = {COMB:None,
                        CRT_VALUE:None,
                        RSS:None}
            eval_comb = comb.copy()
            # ... per ogni grado
            for _ in range(n):
                # riduco il grado della variabile corrente
                eval_comb[i] -= 1
                if np.count_nonzero(eval_comb) == 0:
                    continue
                # costruisco il dataset con la combinazione corrente ...
                X_eval = build_dataset(eval_comb, X)
                # valuto il modello
                result = eval(X_eval, Y)
                # Aggiorno il migliore del polinomio
                if best_deg[CRT_VALUE] == None or (minimize and result[criterion] < best_deg[CRT_VALUE]) or (maximize and result[criterion] > best_deg[CRT_VALUE]):
                    best_deg[COMB] = eval_comb.copy()
                    best_deg[CRT_VALUE] = result[criterion]
                    best_deg[RSS] = result[RSS]
                n_tested += 1
            # Aggiorno il migliore dell'iterazione
            if best_iter[RSS] == None or best_deg[RSS] < best_iter[RSS]:
                best_iter[COMB] = best_deg[COMB]
                best_iter[CRT_VALUE] = best_deg[CRT_VALUE]
                best_iter[RSS] = best_deg[RSS]
        # Aggiorno la combinazione corrente
        comb = best_iter[COMB]
        # Aggiorno il migliore globale
        if best_iter[CRT_VALUE] != None:
            if best[CRT_VALUE] == None or (minimize and best_iter[CRT_VALUE] < best[CRT_VALUE]) or (maximize and best_iter[CRT_VALUE] > best[CRT_VALUE]):
                best[COMB] = best_iter[COMB]
                best[CRT_VALUE] = best_iter[CRT_VALUE]
                best[RSS] = best_iter[RSS]
    return best, n_tested

def stepwise_selection(X: pd.DataFrame, Y: pd.Series, 
                        eval : Callable[[pd.DataFrame, pd.Series], dict], 
                        criterion: str, 
                        n: int = 1, 
                        minimize: bool = True):
    maximize = not minimize
    # Miglior risultato
    best = {COMB:None,
            CRT_VALUE:None,
            RSS:None}
    # Numero di variabili
    n_var = X.columns.size
    # Numero di modelli testati
    n_tested = 0
    # Combinazione corrente
    comb = np.zeros(shape = (n_var, ), dtype=int)

    while np.count_nonzero(comb) != n_var :
        # ... resetto il miglior risultato dell'iterazione
        best_iter = {COMB:None,
                     CRT_VALUE:None,
                     RSS:None}
        for i in range(n_var):
            if comb[i] > 0:
                continue
            # ... resetto il miglior risultato del polinomio
            best_deg = {COMB:None,
                        CRT_VALUE:None,
                        RSS:None}
            eval_comb = comb.copy()
            # ... per ogni grado
            for _ in range(n):
                # aumento il grado della variabile corrente
                eval_comb[i] += 1
                # costruisco il dataset con la combinazione corrente ...
                X_eval = build_dataset(eval_comb, X)
                # valuto il modello
                result = eval(X_eval, Y)
                # Aggiorno il migliore del polinomio
                if best_deg[CRT_VALUE] == None or (minimize and result[criterion] < best_deg[CRT_VALUE]) or (maximize and result[criterion] > best_deg[CRT_VALUE]):
                    best_deg[COMB] = eval_comb.copy()
                    best_deg[CRT_VALUE] = result[criterion]
                    best_deg[RSS] = result[RSS]
                n_tested += 1
            # Aggiorno il migliore dell'iterazione
            if best_iter[RSS] == None or best_deg[RSS] < best_iter[RSS]:
                best_iter[COMB] = best_deg[COMB]
                best_iter[CRT_VALUE] = best_deg[CRT_VALUE]
                best_iter[RSS] = best_deg[RSS]
        # Aggiorno la combinazione corrente
        comb = best_iter[COMB]
        #
        # --- BACKWARD STEP ---
        # 
        if np.count_nonzero(comb) > 2:
            for i in range(n_var):
                if comb[i] == 0:
                    continue
                # ... resetto il miglior risultato del polinomio
                best_deg = {COMB:None,
                            CRT_VALUE:None,
                            RSS:None}
                eval_comb = comb.copy()
                # ... per ogni grado
                for _ in range(n):
                    # riduco il grado della variabile corrente
                    eval_comb[i] -= 1
                    if eval_comb.sum() == 0:
                        continue
                    # costruisco il dataset con la combinazione corrente ...
                    X_eval = build_dataset(eval_comb, X)
                    # valuto il modello
                    result = eval(X_eval, Y)
                    # Aggiorno il migliore del polinomio
                    if best_deg[CRT_VALUE] == None or (minimize and result[criterion] < best_deg[CRT_VALUE]) or (maximize and result[criterion] > best_deg[CRT_VALUE]):
                        best_deg[COMB] = eval_comb.copy()
                        best_deg[CRT_VALUE] = result[criterion]
                        best_deg[RSS] = result[RSS]
                    n_tested += 1
                # Aggiorno il migliore dell'iterazione
                if best_iter[RSS] == None or best_deg[RSS] < best_iter[RSS]:
                    best_iter[COMB] = best_deg[COMB]
                    best_iter[CRT_VALUE] = best_deg[CRT_VALUE]
                    best_iter[RSS] = best_deg[RSS]
            # Aggiorno la combinazione corrente
            comb = best_iter[COMB]
        # Aggiorno il migliore globale
        if best[CRT_VALUE] == None or (minimize and best_iter[CRT_VALUE] < best[CRT_VALUE]) or (maximize and best_iter[CRT_VALUE] > best[CRT_VALUE]):
            best[COMB] = best_iter[COMB]
            best[CRT_VALUE] = best_iter[CRT_VALUE]
            best[RSS] = best_iter[RSS]
    return best, n_tested

