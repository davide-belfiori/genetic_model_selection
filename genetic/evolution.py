'''
    Questo file contiene la definizione delle calssi
    di implementazione di un ALGORITMO GENETICO, sia
    in versione sequenziale che parallela.
'''

# --------------
# --- IMPORT ---
# --------------

import random
from typing import Callable
from genetic.population import Element, Population, roulette_wheel_selection
from genetic.problem import *
import concurrent.futures
import numpy as np

# -----------------
# --- ECCEZIONI ---
# -----------------

class UnipmlementedException(Exception):

    def __init__(self, message: str = "Method or Strategy not Implemented.") -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> str:
        return self.message

# --------------
# --- CLASSI ---
# --------------

class GeneticSolver():
    """
        Implementazione di un algoritmo genetico.
    """
    def __init__(self, problem: GAProblem, pop_size: int = 20, p_cross: float = 0.9, p_mut: float = 0.05, selection_strategy = "roulette_wheel", pop_update_strategy: str = "generational", use_elit: bool = True, minimize: bool = False) -> None:
        """
            Costruttore di un algoritmo genetico.

            Parametri:
            - problem: problema da ottimizzare
            - pop_size: dimensione della popolazione iniziale
            - p_cross: probailità iniziale di crossover tra due elementi della popolazione [min = 0.0, max = 1.0]
            - p_mut: probalilità iniziale di mutazione di un elemento [min = 0.0, max = 1.0]
            - selection_strategy: strategia di selezione delle coppie di elementi.
            - pop_update_strategy: strategia di aggiornamento della popolazione, può essere 'generational' (gli elementi figli sostituiscono i genitori) oppure 'best_n' (la generazione successiva è composta dai migliori n elementi tra genitori e figli)
            - use_elit: se `pop_update_strategy` = 'generational', allora `use_elit` indica se utilizzare o meno l'elitismo
            - minimize: indica se la funzione obiettivo del problema deve essere minimizzata o meno (default = False)
        """
        self.problem = problem
        self.pop_size = pop_size
        self.init_p_cross = p_cross
        self.init_p_mut = p_mut
        self.p_cross = self.init_p_cross
        self.p_mut = self.init_p_mut
        self.selection_strategy = selection_strategy
        self.pop_update_strategy = pop_update_strategy
        self.use_elit = use_elit
        self.minimize = minimize

        # Miglior elemento calcolato
        self.best: Element = None
        # Numero di generazioni dall'ultimo miglioramento
        self.last_update = 0
        # Numero di epoca
        self.epoch = 0
        # Scheduler della probalilità di mutazione
        self.p_mut_scheduler = lambda p_mut, epoch, best, last_update, fit_history : p_mut
        # Scgeduler della probailità di crossover
        self.p_cross_scheduler = lambda p_cross, epoch, best, last_update, fit_history: p_cross
        # Cronologia della Fitness del miglior individuo
        self.fit_history = []

    def use_p_mut_scheduler(self, scheduler: Callable[[float, int, Element, int, list], float]):
        '''
            Imposta lo scheduler della probalilità di mutazione
        '''
        self.p_mut_scheduler = scheduler

    def use_p_cross_scheduler(self, scheduler: Callable[[float, int, Element, int, list], float]):
        '''
            Imposta lo scheduler della probabilità di crossover
        '''
        self.p_cross_scheduler = scheduler

    def is_best(self, elm: Element) -> bool:
        """
            Restituisce True se l'elemento dato è migliore del miglior elemento corrente, False altrimenti.
        """
        if elm == None:
            return False
        if self.minimize:
            return self.best == None or elm.fitness < self.best.fitness
        return self.best == None or elm.fitness > self.best.fitness

    def update_best(self):
        """
            Aggiorna il miglior elemento calcolato con il miglior elemento della popolazione corrente.
        """
        if self.pop != None:
            pop_best = self.pop.first()
            if self.is_best(pop_best):
                self.best = pop_best
                self.last_update = 0
            else:
                self.last_update += 1
                    
    def selection(self) -> 'list':
        """
            Calcola la lista di coppie di elementi da far riprodurre in base al metodo di selezione specificato.
        """
        if self.selection_strategy == "roulette_wheel":
            return roulette_wheel_selection(population = self.pop, minimize = self.minimize)
        else:
            raise UnipmlementedException(message=self.selection_strategy + " Selection Strategy not implemented.")

    def crossover(self, mating_pool: 'list[tuple[Element]]') -> list:
        """
            Calcola la generazione successiva applicando il crossover ad un insieme di coppie
        """
        next_gen = [] # generazione successiva
        # Per ogni coppia nel mating pool ...
        for (par1, par2) in mating_pool:
            # ... estrae un numero casuale tra 0 ed 1 ...
            rand = random.random()
            # ... se il numero è inferiore alla probabilità di crossover ...
            if rand < self.p_cross:
                # ... genera una coppia di figli a partire dalla coppia di genitori
                sol1, sol2 = self.problem.crossover(par1.solution, par2.solution)
                ch1 = Element(solution = sol1, fitness = None)
                ch2 = Element(solution = sol2, fitness = None)
            else:
                # ... altrimenti copia la coppia di genitori
                ch1 = Element(solution = self.problem.copy(par1.solution), fitness = par1.fitness)
                ch2 = Element(solution = self.problem.copy(par2.solution), fitness = par2.fitness)
            # Aggiunge la nuova coppia alla generazione successiva
            next_gen.extend([ch1, ch2])

        return next_gen

    def mutate(self, elements: 'list[Element]'):
        """
            Applica l'operatore di mutazione ad una lista di elementi. 
            Ogni elemento nella lista ha una probalilità `p_mut` di mutare
        """
        # Per ogni individuo della popolazione ...
        for i, elm in enumerate(elements):
            # ... estrae un numero casuale e, se questo è inferiore alla probabilità di mutazione ...
            if random.random() < self.p_mut:
                # ... genera il nuovo individuo mutato
                solution = self.problem.mutate(elm.solution)
                elements[i] = Element(solution, None)

    def update_population(self, next_gen: 'list[Element]'):
        """
            Aggiorna la popolazione secondo la strategia scelta.
        """
        if self.pop_update_strategy == "generational":
            # Aggiornamento generazionale
            self.pop.update(next_gen, self.use_elit)
        elif self.pop_update_strategy == "best_n":
            # Mantenimento dei migliori
            self.pop.extend(next_gen)
            self.pop.resize(self.pop_size)
        else:
            raise UnipmlementedException(message = self.pop_update_strategy + " Update Strategy not implemented.")

    def run(self, epochs: int = 100, verbose: bool = True, seed:int = None) -> Element:
        """
            Esegue `epochs` iterazioni dell'algoritmo e restituisce il miglior elemento calcolato.
        """
        # Reset dei parametri
        self.epoch = 0
        self.last_update = 0
        self.fit_history = []
        self.p_cross = self.init_p_cross
        self.p_mut = self.init_p_mut
        self.best = None

        if seed != None:
            random.seed(seed)
            np.random.seed(seed)

        # Popolazione iniziale
        if verbose:
            print("Inizializzazione popolazione")
        elements = list(map(lambda sol : Element(sol, None), self.problem.init_pop(self.pop_size)))
        # Valutazione della pop. iniziale
        self.problem.evaluate_population(elements)
        self.pop = Population(elements, descending = not self.minimize)

        if verbose:
            print("Calcolo miglior individuo")
        # Aggiornamento miglior individuo
        self.update_best()
        self.fit_history.append(self.best.fitness)

        for epoch in range(epochs):
            self.epoch = epoch
            if verbose:
                self.print_epoch_summary()
            # Callback inizio epoca
            self.problem.on_epoch_start(self.epoch, self.best, self.last_update, self.fit_history)

            # Selezione
            mating_pool = self.selection()
            # Crossover
            next_gen = self.crossover(mating_pool)
            # Mutazione
            self.mutate(next_gen)
            # Valutazione
            self.problem.evaluate_population(next_gen)
            # Aggiornamento popolazione
            self.update_population(next_gen)
            # Aggiornamento miglior individuo
            self.update_best()
            self.fit_history.append(self.best.fitness)
            # Aggiornamento probabilità mutazione e crossover
            self.p_mut = self.p_mut_scheduler(self.p_mut, self.epoch, self.best, self.last_update, self.fit_history)
            self.p_cross = self.p_cross_scheduler(self.p_cross, self.epoch, self.best, self.last_update, self.fit_history)
            
            # Callback fine epoca
            self.problem.on_epoch_end(self.epoch, self.best, self.last_update, self.fit_history)

        self.epoch = epochs
        if verbose:
            self.print_epoch_summary()

        return self.best

    def print_epoch_summary(self):
        '''
            Stampa le statistiche dell'epoca in corso
        '''
        print("Generazione", self.epoch, 
            ":\n    Fitness miglior individuo =", self.best.fitness, 
            "\n    Generazioni dall'ultimo miglioramento =", self.last_update, 
            "\n    Media Fitness dei primi 10 elementi =", self.pop.mean(10),
            "\n    Probabilità crossover =", self.p_cross, 
            "\n    Probabilità mutazione =", self.p_mut)


class MultiThreadGeneticSolver():
    """
        Implementazione di un algoritmo genetico multi-thread.
    """
    def __init__(self, problem: MultiThreadGAProblem, pop_size: int = 20, p_cross: float = 0.9, p_mut: float = 0.05, selection_strategy = "roulette_wheel", pop_update_strategy: str = "generational", use_elit: bool = True, minimize: bool = False, n_jobs: int = 2) -> None:
        """
            Costruttore di un algoritmo genetico multi-thread.

            Parametri:
            - problem: problema da ottimizzare
            - pop_size: dimensione della popolazione iniziale
            - p_cross: probailità iniziale di crossover tra due elementi della popolazione [min = 0.0, max = 1.0]
            - p_mut: probalilità iniziale di mutazione di un elemento [min = 0.0, max = 1.0]
            - selection_strategy: strategia di selezione delle coppie di elementi.
            - pop_update_strategy: strategia di aggiornamento della popolazione, può essere 'generational' (gli elementi figli sostituiscono i genitori) oppure 'best_n' (la generazione successiva è composta dai migliori n elementi tra genitori e figli)
            - use_elit: se `pop_update_strategy` = 'generational', allora `use_elit` indica se utilizzare o meno l'elitismo
            - minimize: indica se la funzione obiettivo del problema deve essere minimizzata o meno (default = False)
            - n_jobs: numero di esecuzioni parallele
        """
        self.problem = problem
        self.pop_size = pop_size
        self.init_p_cross = p_cross
        self.init_p_mut = p_mut
        self.p_cross = self.init_p_cross
        self.p_mut = self.init_p_mut
        self.selection_strategy = selection_strategy
        self.pop_update_strategy = pop_update_strategy
        self.use_elit = use_elit
        self.minimize = minimize

        self.n_jobs = n_jobs
        # Occorre definire la dimensione dei batch di coppie di elementi da processare in parallelo
        # half_pop_size corrisponde alla metà della dimensione della popolazione, allo stesso tempo è pari al numero di elementi nel mating pool di ogni generazione
        self.half_pop_size = self.pop_size // 2
        # Dimensione di ogni sotto-insieme di individui da elaborare in parallelo
        self.pop_batch_size = self.half_pop_size // self.n_jobs
        if self.pop_batch_size < 1:
            # TODO: definire errore
            raise ValueError()

        # Miglior elemento calcolato
        self.best: Element = None
        # Numero di generazioni dall'ultimo miglioramento
        self.last_update = 0
        # Numero di epoca
        self.epoch = 0
        # Scheduler della probalilità di mutazione
        self.p_mut_scheduler = lambda p_mut, epoch, best, last_update, fit_history : p_mut
        # Scgeduler della probailità di crossover
        self.p_cross_scheduler = lambda p_cross, epoch, best, last_update, fit_history: p_cross
        # Cronologia della Fitness del miglior individuo
        self.fit_history = []

    def use_p_mut_scheduler(self, scheduler: Callable[[float, int, Element, int, list], float]):
        '''
            Imposta lo scheduler della probalilità di mutazione
        '''
        self.p_mut_scheduler = scheduler

    def use_p_cross_scheduler(self, scheduler: Callable[[float, int, Element, int, list], float]):
        '''
            Imposta lo scheduler della probabilità di crossover
        '''
        self.p_cross_scheduler = scheduler

    def is_best(self, elm: Element) -> bool:
        """
            Restituisce True se l'elemento dato è migliore del miglior elemento corrente, False altrimenti.
        """
        if elm == None:
            return False
        if self.minimize:
            return self.best == None or elm.fitness < self.best.fitness
        return self.best == None or elm.fitness > self.best.fitness

    def update_best(self):
        """
            Aggiorna il miglior elemento calcolato con il miglior elemento della popolazione corrente.
        """
        if self.pop != None:
            pop_best = self.pop.first()
            if self.is_best(pop_best):
                self.best = pop_best
                self.last_update = 0
            else:
                self.last_update += 1
                    
    def selection(self) -> 'list[tuple[Element, Element]]':
        """
            Calcola la lista di coppie di elementi da far riprodurre in base al metodo di selezione specificato.
        """
        if self.selection_strategy == "roulette_wheel":
            return roulette_wheel_selection(population = self.pop, minimize = self.minimize)
        else:
            raise UnipmlementedException(message=self.selection_strategy + " Selection Strategy not implemented.")

    def crossover(self, mating_pool: 'list[tuple[Element, Element]]', thread_idx: int) -> list:
        """
            Calcola la generazione successiva applicando il crossover ad un insieme di coppie
        """
        next_gen = [] # generazione successiva
        # Per ogni coppia nel mating pool ...
        for (par1, par2) in mating_pool:
            # ... estrae un numero casuale tra 0 ed 1 ...
            rand = random.random()
            # ... se il numero è inferiore alla probabilità di crossover ...
            if rand < self.p_cross:
                # ... genera una coppia di figli a partire dalla coppia di genitori
                sol1, sol2 = self.problem.crossover(sol1 = par1.solution, sol2 = par2.solution, thread_idx = thread_idx)
                ch1 = Element(solution = sol1, fitness = None)
                ch2 = Element(solution = sol2, fitness = None)
            else:
                # ... altrimenti copia la coppia di genitori
                ch1 = Element(solution = self.problem.copy(sol = par1.solution, thread_idx = thread_idx), fitness = par1.fitness)
                ch2 = Element(solution = self.problem.copy(sol = par2.solution, thread_idx = thread_idx), fitness = par2.fitness)
            # Aggiunge la nuova coppia alla generazione successiva
            next_gen.extend([ch1, ch2])

        return next_gen

    def mutate(self, elements: 'list[Element]', thread_idx: int):
        """
            Applica l'operatore di mutazione ad una lista di elementi. 
            Ogni elemento nella lista ha una probalilità `p_mut` di mutare
        """
        # Per ogni individuo della popolazione ...
        for i, elm in enumerate(elements):
            # ... estrae un numero casuale e, se questo è inferiore alla probabilità di mutazione ...
            if random.random() < self.p_mut:
                # ... genera il nuovo individuo mutato
                solution = self.problem.mutate(sol = elm.solution, thread_idx = thread_idx)
                elements[i] = Element(solution, None)

    def update_population(self, next_gen: 'list[Element]'):
        """
            Aggiorna la popolazione secondo la strategia scelta.
        """
        if self.pop_update_strategy == "generational":
            # Aggiornamento generazionale
            self.pop.update(next_gen, self.use_elit)
        elif self.pop_update_strategy == "best_n":
            # Mantenimento dei migliori
            self.pop.extend(next_gen)
            self.pop.resize(self.pop_size)
        else:
            raise UnipmlementedException(message = self.pop_update_strategy + " Update Strategy not implemented.")

    def init_evaluation_task(self, pop_batch: 'list[Element]', thread_idx: int):
        '''
            Task di valutazione di un soto-insieme di individui
        '''
        self.problem.evaluate_population(pop_batch, thread_idx)
        return pop_batch

    def run_epoch_task(self, mating_pool: 'list[tuple[Element, Element]]', thread_idx: int, seed: int = None) -> 'list[Element]':
        '''
            Task di Crossover, Mutazione e Valutazione di un sotto-insieme di individui
        '''
        if seed != None:
            random.seed(seed)
            np.random.seed = (seed)
        #   Crossover
        next_gen = self.crossover(mating_pool, thread_idx)
        #   Mutazione
        self.mutate(next_gen, thread_idx)
        #   Valutazione
        self.problem.evaluate_population(next_gen, thread_idx)
        return next_gen

    def run(self, epochs: int = 100, verbose: bool = True, seed: int = None) -> Element:
        """
            Esegue `epochs` iterazioni dell'algoritmo e restituisce il miglior elemento calcolato.
        """
        # Reset dei parametri
        self.epoch = 0
        self.last_update = 0
        self.fit_history = []
        self.p_cross = self.init_p_cross
        self.p_mut = self.init_p_mut
        self.best = None

        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
        
        if verbose:
            print("Inizializzazione popolazione")
        # Generazione degli individui iniziali
        elements = list(map(lambda sol : Element(sol, None), self.problem.init_pop(self.pop_size)))
        # Valutazione parallela degli individui iniziali:
        #   - Calcolo della dimensione di un sotto-insieme di individui
        init_batch_size = self.pop_size // self.n_jobs
        #   - Lista di sotto-insiemi
        batch_list = [elements[i:i + init_batch_size] for i in range(0, self.pop_size, init_batch_size)]
        #   - Popolazione iniziale
        pop = []
        #   - Valutazione parallela
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for i in range(self.n_jobs):
                futures.append(executor.submit(self.init_evaluation_task, pop_batch=batch_list[i], thread_idx=i))
            for future in concurrent.futures.as_completed(futures):
                pop.extend(future.result())
        # Composizione della popolazione iniziale
        self.pop = Population(pop, descending = not self.minimize)

        if verbose:
            print("Calcolo miglior individuo")
        # Aggiornamento individuo migliore
        self.update_best()
        self.fit_history.append(self.best.fitness)

        for epoch in range(epochs):
            self.epoch = epoch
            if verbose:
                self.print_epoch_summary()
            # Callback inizio epoca
            self.problem.on_epoch_start(self.epoch, self.best, self.last_update, self.fit_history)

            # Selezione (ATTENZIONE: la SELEZIONE NON può essere PARALLELIZZATA!)
            mating_pool = self.selection()
            # Calcolo dei sotto-insiemi di coppie di elementi da processare in parallelo
            m_pool_batch = [mating_pool[i:i + self.pop_batch_size] for i in range(0, self.half_pop_size, self.pop_batch_size)]
            # Calcolo parallelo della generazione successiva
            next_gen = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                for i in range(self.n_jobs):
                    futures.append(executor.submit(self.run_epoch_task, mating_pool=m_pool_batch[i], thread_idx=i))
                for future in concurrent.futures.as_completed(futures):
                    next_gen.extend(future.result())

            # Aggiornamento popolazione
            self.update_population(next_gen)
            # Aggiornamento miglior individuo
            self.update_best()
            self.fit_history.append(self.best.fitness)
            # Aggiornamento probabilità mutazione e crossover
            self.p_mut = self.p_mut_scheduler(self.p_mut, self.epoch, self.best, self.last_update, self.fit_history)
            self.p_cross = self.p_cross_scheduler(self.p_cross, self.epoch, self.best, self.last_update, self.fit_history)

            # Callback fine epoca
            self.problem.on_epoch_end(self.epoch, self.best, self.last_update, self.fit_history)

        self.epoch = epochs
        if verbose:
            self.print_epoch_summary()

        return self.best

    def print_epoch_summary(self):
        '''
            Stampa le statistiche dell'epoca in corso
        '''
        print("Generazione", self.epoch, 
              ":\n    Fitness miglior individuo =", self.best.fitness, 
              "\n    Generazioni dall'ultimo miglioramento =", self.last_update, 
              "\n    Media Fitness dei primi 10 elementi =", self.pop.mean(10),
              "\n    Probabilità crossover =", self.p_cross, 
              "\n    Probabilità mutazione =", self.p_mut)
