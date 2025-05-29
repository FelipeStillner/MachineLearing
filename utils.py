import pandas as pd
import numpy as np
from collections import Counter
import math


def get_entropy(labels: pd.Series) -> float :
    total: int = len(labels)
    contagem: Counter = Counter(labels)
    entropy_calc: float = 0.0

    for count in contagem.values():
        if count != 0:
            prob: float = count/total
            entropy_calc += -prob *math.log2(prob)

    return entropy_calc
    
def get_info_gain(df: pd.DataFrame, atribute: str, target: str) -> float :
    entropy_total: float = get_entropy(df[target])
    values: np.ndarray = df[atribute].unique()
    weighted: float = 0
    
    for v in values :
        subset: pd.DataFrame = df[df[atribute] == v]
        weighted +=(len(subset)/len(df))*get_entropy(subset[target])

    return entropy_total - weighted