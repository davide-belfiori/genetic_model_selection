'''
    Questo file contiene la definizione delle calssi
    di implementazione di una POPOLAZIONE di individui, 
    più le funzioni di selezione del MATING POOL.
'''

# --------------
# --- IMPORT ---
# --------------

from typing import Iterator, Sized
import random

# --------------
# --- CLASSI ---
# --------------

class Element():
    '''
        Definisce un individuo di una popolazione
    '''
    def __init__(self, solution: object, fitness: float, data: dict = None) -> None:
        '''
            Costruttore di un individuo.

            Parametri:
            - `solution`: soluzione rappresentata dall'individuo
            - `fitness`: valore di bontà della soluzione
            - `data`: dati aggiuntivi
        '''
        self.solution = solution
        self.fitness = fitness
        self.data = data

    def __str__(self) -> str:
        return "Solution : " + str(self.solution) + "\n Fitness : " + str(self.fitness)


class Population(Iterator, Sized):
    '''
        Definisce una pololazione di individui ordinati in base al valore di fitness.
    '''
    def __init__(self, elements : 'list[Element]' = [], descending: bool = False) -> None:
        '''
            Costruttore di una popolazione.

            Parametri:
            - `elements`: lista di individui.
            - `decsending`: quando è uguale a `True` indica che la popolazione deve essere ordinata in 
                            ordine decrescente rispetto al valore di fitness.
        '''
        self.descending = descending
        self.elements = sorted(elements, key = lambda x : x.fitness, reverse = self.descending)

    def pop_size(self):
        '''
            Restituisce il numero di individui della popolazione.
        '''
        if self.elements == None:
            return 0
        return len(self.elements)

    def first(self):
        '''
            Restituisce il primo elemento della popolazione.

            Restituisce `None` se la popolazione è vuota.
        '''
        if len(self.elements) == 0:
            return None
        return self.elements[0]

    def last(self):
        '''
            Restituisce l'ultimo elemento della popolazione.

            Restituisce `None` se la popolazione è vuota.
        '''
        if len(self.elements) == 0:
            return None
        return self.elements[-1]

    def mean(self, n: int = None):
        """
            Restituisce la media dei valori di fitness dei primi `n` elementi.

            Se `n = None`, la media è calcolata su tutti gli elementi della popolazione.
        """
        if len(self.elements) == 0:
            return None
        if n == None or n >= len(self.elements):
            n = self.pop_size()
        fit_list = list(map(lambda elm : elm.fitness, self.elements[:n]))
        return sum(fit_list) / n

    def resize(self, n: int):
        '''
            Mantiene nella popolazione solo i primi `n` individui.
        '''
        if n > 0:
            self.elements = self.elements[:n]

    def add(self, element: Element):
        '''
            Aggiunge un individuo alla popolazione, rispettando l'ordine di fitness.
        '''
        for i in range(len(self.elements)):
            if (self.descending and self.elements[i].fitness <= element.fitness) or \
               (not self.descending and self.elements[i].fitness >= element.fitness):
                self.elements.insert(i, element)
                return
        self.elements.append(element)

    def extend(self, batch: 'list[Element]'):
        '''
            Aggiunge una lista di individui alla popolazione, rispettando l'ordine di fitness.
        '''
        for elm in batch:
            self.add(elm)

    def update(self, next_gen: 'list[Element]', use_elit: bool = False):
        '''
            Sostituisce l'insieme di individui corrente con un nuovo insieme.

            Applica l'elitismo se `use_elit = True`.
        '''
        if next_gen != None and len(next_gen) > 0:
            old_best = self.first()
            next_gen = sorted(next_gen, key = lambda x : x.fitness, reverse = self.descending)
            self.elements = next_gen
            if use_elit:
                if (self.descending and old_best.fitness >= next_gen[-1].fitness) or \
                   (not self.descending and old_best.fitness <= next_gen[-1].fitness):
                   self.add(old_best)
                   self.elements = self.elements[:-1]

    def __iter__(self) -> Iterator[Element]:
        self.iter_idx = 0
        return self

    def __next__(self) -> Element:
        if self.iter_idx >= len(self.elements):
            self.iter_idx = 0
            raise StopIteration
        to_return = self.elements[self.iter_idx]
        self.iter_idx += 1
        return to_return

    def __getitem__(self, x):
        return self.elements.__getitem__(x)

    def __len__(self) -> int:
        return self.pop_size()

# ----------------
# --- FUNZIONI ---
# ----------------

def roulette_wheel_selection(population: 'Population | list[Element]', minimize: bool = False) -> 'list[tuple[Element, Element]]':
    """
        Calcola l'insieme di coppie di elementi di una popolazione da far riprodurre, mediante la tecnica Roulette Wheel.
    """
    mating_pool = []
    if minimize:
        fit = [1.0 / (elm.fitness + 1) for elm in population]
    else:
        fit = [elm.fitness for elm in population]
    sum_fit = sum(fit)
    # Probalilità di selezione
    prob = [f / sum_fit for f in fit]
    # Formazione delle coppie
    for _ in range(len(population) // 2):
        mating_pool.append((population[roulette_wheel(prob)],
                            population[roulette_wheel(prob)]))
    return mating_pool

def roulette_wheel (prob_list : "list[float]") -> int:
        """
            Restituisce l'indice di un elemento di una lista di probailità
        """
        rand = random.random()
        j = 0
        while prob_list[j] < rand:
            rand = rand - prob_list[j]
            j += 1
        return j