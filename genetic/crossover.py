import random
from bitarray import bitarray, util
import numpy as np

# -----------------
# --- ECCEZIONI ---
# -----------------

class PointCrossoverException(Exception):

    def __init__(self, message : str) -> None:
        super().__init__()
        self.message = "Point Crossover Exception: " + message

    def __str__(self) -> str:
        return self.message

class FloatCrossoverException(Exception):

    def __init__(self, message : str) -> None:
        super().__init__()
        self.message = "Float Crossover Exception: " + message

    def __str__(self) -> str:
        return self.message

class ByteCrossoverException(Exception):

    def __init__(self, message : str) -> None:
        self.message = "Byte Crossover Exception: " + message
        super().__init__()

    def __str__(self) -> str:
        return self.message

class TensorShapeException(Exception):

    def __init__(self, message : str) -> None:
        self.message = "Tensor Shape Exception: " + message
        super().__init__()

    def __str__(self) -> str:
        return self.message

# ----------------
# --- FUNZIONI ---
# ----------------

def n_point_crossover(x: list, y: list, n: int = 1) :
    """
        Crossover ad n punti tra due liste di oggetti.
    """
    minlen = min(len(x), len(y))
    if n < 0 or n >= minlen:
        raise PointCrossoverException("Invalid points number, n must be in range (0, length of shorter sequence).")
    points = sorted(random.sample(range(1, minlen), n))
    x1 = []
    y1 = []
    start = 0
    for i, point in enumerate(points):
        if i % 2 == 0:
            x1.extend(x[start:point])
            y1.extend(y[start:point])
        else:
            x1.extend(y[start:point])
            y1.extend(x[start:point])
        start = point

    if n % 2 == 0:
        x1.extend(x[start:])
        y1.extend(y[start:])
    else:
        x1.extend(y[start:])
        y1.extend(x[start:])

    return x1, y1, points


def uniform_crossover(x: list, y: list) :
    """
        Crossover uniforme tra due liste di oggetti.
    """
    minlen = min(len(x), len(y))
    rand_string = [random.randint(0,1) for _ in range(0, minlen)]
    x1 = []
    y1 = []
    for i, k in enumerate(rand_string):
        if k == 0:
            x1.append(x[i])
            y1.append(y[i])
        else: 
            x1.append(y[i])
            y1.append(x[i])
    return x1, y1, rand_string


def float_crossover(x: float, y: float, k_mode = "normal", k_range = (0,1), k_mean_dev = (0,1), _k: float = 0.5) :
    """
        Crossover tra numeri reali. I figli vengono generati come combinazione convessa tra `x` ed `y`:

            x1 = k * x + (1-k) * y

            y1 = (1 - k) * x + k * y 

        Parametri:
        - x : primo genitore
        - y : secondo genitore
        - k_mode : metodo di selezione di k, può essere 'normal', 'uniform' oppure 'const'
        - k_range : se `k_mode` = 'uniform', allora k_range rappresenta l'intervallo di valori all'interno del quale k può essere selezionato
        - k_mead_dev : se `k_mode` = 'normal', allora k_mead_dev rappresenta la coppia (media, dev. standard) della distribuzione normale dal quale k viene generato
        - _k : se `k_mode` = 'const', allora _k rappresenta l'effetivo valore di k
    """
    if k_mode == "normal":
        k = random.normalvariate(k_mean_dev[0], k_mean_dev[1])
    elif k_mode == "uniform":
        k = random.uniform(k_range[0], k_range[1])
    elif k_mode == "const":
        k = _k
    else:
        raise FloatCrossoverException(message = "Invalid k selection mode. Must be 'normal', 'uniform' or 'const'.")

    x1 = k * x + (1 - k) * y
    y1 = (1 - k) * x + k * y

    return x1, y1, k


def byte_crossover(x: int, y: int, type : str = "uniform", n = 1) -> 'tuple[int, int]' :
    """
        Crossover tra due byte espressi come interi senza segno ad 8 bit.

        Parametri:
        - a: primo byte
        - b: secondo byte
        - type: tipo di crossover, può essere 'uniform' nel caso di crossover uniforme, o 'n_point' per il crossover ad n punti
        - n: se `type` = 'n_point', allora n indica il numero di punti di crossover
    """
    if x < 0 or x > 255 or y < 0 or y > 255:
        raise ByteCrossoverException(message="Invalid numeric format, 'x' and 'y' must be in range [0, 255]")

    par1 = bitarray()
    par1.frombytes(int(x).to_bytes(1, 'big'))
    par2 = bitarray()
    par2.frombytes(int(y).to_bytes(1, 'big'))

    ch1 = bitarray()
    ch2 = bitarray()

    if type == 'uniform':
        x1, x2, _ = uniform_crossover(par1, par2)
    elif type == "n_point":
        x1, x2, _ = n_point_crossover(par1, par2, n)
    else:
        raise ByteCrossoverException("Invalid crossover type, must be 'uniform' or 'n_point'.")

    ch1.extend(x1)
    ch2.extend(x2)
    return util.ba2int(ch1), util.ba2int(ch2)


def float_tensor_crossover(X: np.ndarray, Y: np.ndarray, K: np.ndarray = None, k_mode: str = "normal", k_range = (0,1), k_mean_dev = (0,1), _k: float = 0.5):    
    """
        Crossover due tensori di numeri reali.

            X1 = K * X + (1-K) * Y

            Y1 = (1 - K) * X + K * Y

        Parametri:
        - X : primo tensore
        - Y : secondo tensore
        - K : tensore moltiplicativo. se K = None, viene generato un tensore in base alla modalità di selezione scelta
        - k_mode : se K = None, indica il metodo di selezione delle componenti di K
        - k_range : se `k_mode` = 'uniform', allora k_range rappresenta l'intervallo di valori all'interno del quale le componenti di K possono essere selezionate
        - k_mead_dev : se `k_mode` = 'normal', allora k_mead_dev rappresenta la coppia (media, dev. standard) della distribuzione normale dal quale le componenti di K vengono generate
        - _k : se `k_mode` = 'const', allora _k rappresenta l'effetivo valore delle componenti di K
    """
    if (not isinstance(K, np.ndarray) and K == None) or not K.any():
        if k_mode == "normal":
            K = np.random.normal(k_mean_dev[0], k_mean_dev[1], X.shape)
        elif k_mode == "uniform":
            K = np.random.uniform(k_range[0], k_range[1], X.shape)
        elif k_mode == "const":
            K = np.full(X.shape, _k)
        else:
            raise FloatCrossoverException(message = "Invalid k selection mode. Must be 'normal', 'uniform' or 'const'.")

    if (X.shape != Y.shape) or (X.shape != K.shape):
        raise TensorShapeException("Invalid tensor shape.")

    _one = np.ones(X.shape)
    X1 = K * X + (_one - K) * Y
    Y1 = (_one - K) * X + K * Y

    return X1, Y1
