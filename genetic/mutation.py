import random
from bitarray import bitarray, util
import numpy as np

# -----------------
# --- ECCEZIONI ---
# -----------------

class FloatMutationException(Exception):

    def __init__(self, message : str) -> None:
        super().__init__()
        self.message = "Float Mutation Exception: " + message

    def __str__(self) -> str:
        return self.message

class ByteMutationException(Exception):

    def __init__(self, message : str) -> None:
        self.message = "Byte Mutation Exception: " + message
        super().__init__()

    def __str__(self) -> str:
        return self.message

# -----------------
# --- FUNCTIONS ---
# -----------------

def float_mutation(x: float, k_mode = "normal", k_range = (0,1), k_mean_dev = (0,1), _k: float = 1.0, _F = None) -> float:
    """
        Mutazione di un numero reale.

            x1 = _F(x + k)
        
        Parametri:
        - x : elemento da mutare
        - k_mode : metodo di selezione di k, può essere 'normal', 'uniform' oppure 'const'
        - k_range : se `k_mode` = 'uniform', allora k_range rappresenta l'intervallo di valori all'interno del quale k può essere selezionato
        - k_mead_dev : se `k_mode` = 'normal', allora k_mead_dev rappresenta la coppia (media, dev. standard) della distribuzione normale dal quale k viene generato
        - _k : se `k_mode` = 'const', allora _k rappresenta l'effetivo valore di k
        - _F : funzione di trasformazione (float -> float)
    """
    if k_mode == "normal":
        k = random.normalvariate(k_mean_dev[0], k_mean_dev[1])
    elif k_mode == "uniform":
        k = random.uniform(k_range[0], k_range[1])
    elif k_mode == "const":
        k = _k
    else:
        raise FloatMutationException(message = "Invalid k selection mode. Must be 'normal', 'uniform' or 'const'.")

    x1 = x + k
    if _F != None:
        return _F(x1)
    return x1

def byte_mutation(x: int) -> int:
    """
        Mutazione di un byte espresso come intero senza segno ad 8 bit.

        La funzione inverte casualmente un bit di `x`.
    """
    if x < 0 or x > 255:
        raise ByteMutationException(message="Invalid numeric format, 'x' must be in range [0, 255]")
    
    x1 = bitarray()
    x1.frombytes(int(x).to_bytes(1, 'big'))
    # indice del bit da mutare
    mut_idx = random.randint(0, 7)
    if x1[mut_idx] == 0:
        x1[mut_idx] = 1
    else:
        x1[mut_idx] = 0
    
    return util.ba2int(x1)
    

def float_tensor_mutation(X: np.ndarray, strength: float = 0.1, k_mode = "normal", k_range = (0,1), k_mean_dev = (0,1), _k: float = 1.0, _F = None):
    """
        Mutazione di un tensore di numeri reali. Se x è un elemento di `X`, allora:

            x1 = _F(x + k)
        
        Parametri:
        - X : tensore da mutare
        - stregth : probabilità di mutazione di un singolo elemento di `X`
        - k_mode : metodo di selezione di k, può essere 'normal', 'uniform' oppure 'const'
        - k_range : se `k_mode` = 'uniform', allora k_range rappresenta l'intervallo di valori all'interno del quale k può essere selezionato
        - k_mead_dev : se `k_mode` = 'normal', allora k_mead_dev rappresenta la coppia (media, dev. standard) della distribuzione normale dal quale k viene generato
        - _k : se `k_mode` = 'const', allora _k rappresenta l'effetivo valore di k
        - _F : funzione di trasformazione (float -> float)
    """
    X1 = np.copy(X)
    X1 = X1.ravel()
    for i, e in enumerate(X1):
        if random.random() < strength:
            X1[i] = float_mutation(e, k_mode, k_range, k_mean_dev, _k, _F)
    X1 = X1.reshape(X.shape)
    return X1

    
def byte_tensor_mutation(X: np.ndarray, strength: float = 0.1):
    """
        Mutazione di un tensore di byte, espressi come interi senza segno ad 8 bit.

        Parametri:
        - X : tensore da mutare
        - stregth : probabilità di mutazione di un singolo elemento di `X`
    """
    X1 = np.copy(X)
    X1 = X1.ravel()
    for i, e in enumerate(X1):
        if random.random() < strength:
            X1[i] = byte_mutation(e)
    X1 = X1.reshape(X.shape)
    return X1