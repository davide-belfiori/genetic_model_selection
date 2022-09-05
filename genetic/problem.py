'''
    Questo file contiene la definizione delle Calssi Astratte
    per l'impementazione di problemi risolvibili mediante
    Algoritmi Genetici.

    Un generico problema deve specificare:
    - una funzione di generazione di soluzioni
    - una funzione di valutazione delle soluzioni
    - una funzione di crossover tra due soluzioni
    - una funzione di mutazione di una soluzione
    - una funzione di copia di una soluzione
'''

# --------------
# --- IMPORT ---
# --------------

from genetic.population import Element

# --------------
# --- CLASSI ---
# --------------

class GAProblem():

    """
        Classe generica per l'implementazione di un problema di ottimizzazione.
    """

    def __init__(self) -> None:
        """
            Costruttore del problema.
        """
        pass

    def init_pop(self, n: int) -> list:
        """
            Genera una popolazione iniziale di `n` soluzioni.
        """
        pass

    def evaluate(self, elm: Element) -> None:
        """
            Calcola il valore di fitness associato ad un Elemento, modifica l'oggetto originale.
        """
        pass

    def evaluate_population(self, pop: 'list[Element]') -> None:
        """
            Calcola il valore di fitness per una popolazione di soluzioni, modifica la lista originale.
        """
        for elm in pop:
            self.evaluate(elm)

    def crossover(self, sol1: object, sol2: object) -> 'list | tuple':
        """
            Restituisce il risultato del crossover tra due soluzioni.
        """
        pass

    def mutate(self, sol: object) -> object:
        """
            Restituisce il risultato della mutazione di una soluzione.
        """
        pass

    def copy(self, sol: object) -> object:
        """
            Restituisce la copia di una soluzione.
        """
        return sol

    def on_epoch_start(self, epoch: int, best: Element, last_best_update: list, fit_history: list):
        """
            Metodo richiamato all'inizio di ogni epoca.
        """
        pass

    def on_epoch_end(self, epoch: int, best: Element, last_best_update: list, fit_history: list):
        """
            Metodo richiamato al termine di ogni epoca.
        """
        pass

class MultiThreadGAProblem():

    """
        Classe generica per l'implementazione di un problema di ottimizzazione multi thread.
    """

    def __init__(self) -> None:
        """
            Costruttore del problema.
        """
        pass

    def init_pop(self, n: int) -> list:
        """
            Genera una popolazione iniziale di `n` soluzioni.
        """
        pass

    def evaluate(self, elm: Element, thread_idx: int = 0) -> None:
        """
            Calcola il valore di fitness associato ad un Elemento, modifica l'oggetto originale.

            Parametri:
            - elm: elemento da valutare
            - thead_idx: indice del thread di di esecuzione
        """
        pass

    def evaluate_population(self, pop: 'list[Element]', thread_idx: int = 0) -> None:
        """
            Calcola il valore di fitness per una popolazione di soluzioni, modifica la lista originale.

            Parametri:
            - pop: lista di elementi da valutare
            - thead_idx: indice del thread di di esecuzione
        """
        for elm in pop:
            self.evaluate(elm, thread_idx)

    def crossover(self, sol1: object, sol2: object, thread_idx: int = 0) -> 'list | tuple':
        """
            Restituisce il risultato del crossover tra due soluzioni.
        """
        pass

    def mutate(self, sol: object, thread_idx: int = 0) -> object:
        """
            Restituisce il risultato della mutazione di una soluzione.
        """
        pass

    def copy(self, sol: object, thread_idx: int = 0) -> object:
        """
            Restituisce la copia di una soluzione.
        """
        return sol

    def on_epoch_start(self, epoch: int, best: Element, last_best_update: list, fit_history: list):
        """
            Metodo richiamato all'inizio di ogni epoca.
        """
        pass

    def on_epoch_end(self, epoch: int, best: Element, last_best_update: list, fit_history: list):
        """
            Metodo richiamato al termine di ogni epoca.
        """
        pass
    