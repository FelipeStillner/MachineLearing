import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Any, Dict, Union
from random import sample, randint
import matplotlib.pyplot as plt

def gini_impurity(y: pd.Series) -> float:
    counts = Counter(y)
    total = len(y)
    impurity = 1.0
    for count in counts.values():
        prob = count / total
        impurity -= prob ** 2
    return impurity

def best_split(df: pd.DataFrame, features: List[str], target: str, max_features: int) -> Dict[str, Any]:
    best_attr = None
    best_thresh = None
    best_gain = -1
    best_splits = None
    features_sample = sample(features, min(max_features, len(features)))
    
    current_impurity = gini_impurity(df[target])
    
    for attr in features_sample:
        values = df[attr].unique()
        for val in values:
            left = df[df[attr] <= val]
            right = df[df[attr] > val]
            if len(left) == 0 or len(right) == 0:
                continue
            p_left = len(left) / len(df)
            p_right = len(right) / len(df)
            gain = current_impurity - (p_left * gini_impurity(left[target]) + p_right * gini_impurity(right[target]))
            if gain > best_gain:
                best_attr = attr
                best_thresh = val
                best_gain = gain
                best_splits = (left, right)
                
    return {'attribute': best_attr, 'threshold': best_thresh, 'splits': best_splits}

def build_tree(df: pd.DataFrame, features: List[str], target: str, max_depth: int, max_features: int, depth: int=0) -> Any:
    if len(df[target].unique()) == 1 or len(df) < 2 or depth >= max_depth:
        return df[target].mode()[0]
    
    split = best_split(df, features, target, max_features)
    if split['attribute'] is None:
        return df[target].mode()[0]
    
    left = build_tree(split['splits'][0], features, target, max_depth, max_features, depth + 1)
    right = build_tree(split['splits'][1], features, target, max_depth, max_features, depth + 1)
    
    return {'attribute': split['attribute'], 'threshold': split['threshold'], 'left': left, 'right': right}

def predict_tree(tree: Any, example: pd.Series) -> Any:
    if not isinstance(tree, dict):
        return tree
    if example[tree['attribute']] <= tree['threshold']:
        return predict_tree(tree['left'], example)
    else:
        return predict_tree(tree['right'], example)

def random_forest_train(df: pd.DataFrame, features: List[str], target: str, n_trees: int, max_features: int, max_depth: int) -> List[Any]:
    trees = []
    for _ in range(n_trees):
        bootstrap = df.sample(frac=1, replace=True, random_state=randint(0, 10000))
        tree = build_tree(bootstrap, features, target, max_depth, max_features)
        trees.append(tree)
    return trees

def random_forest_predict(trees: List[Any], example: pd.Series) -> Any:
    votes = [predict_tree(tree, example) for tree in trees]
    return Counter(votes).most_common(1)[0][0]

def evaluate_rf(df: pd.DataFrame, features: List[str], target: str, n_trees: int, max_features: int, max_depth: int, test_size: float, n_repeats: int) -> List[float]:
    accuracies = []
    for i in range(n_repeats):
        train_df = df.sample(frac=1 - test_size, random_state=i)
        test_df = df.drop(train_df.index)
        trees = random_forest_train(train_df, features, target, n_trees, max_features, max_depth)
        correct = 0
        for _, row in test_df.iterrows():
            pred = random_forest_predict(trees, row)
            if pred == row[target]:
                correct += 1
        acc = correct / len(test_df)
        accuracies.append(acc)
        print(f"Repetição {i+1} - Acurácia: {acc:.2%}")
    return accuracies

def main():
    df = pd.read_csv('./data.csv')
    features = ['pSist', 'pDiast', 'qPA', 'pulso', 'respiracao']
    target = 'classe'
    n_trees = 20
    max_features = int(np.sqrt(len(features)))
    max_depth = 20  
    
    print("\n--- Random Forest (Tradicional) 80% teste ---")
    acc_80_20 = evaluate_rf(df, features, target, n_trees, max_features, max_depth, test_size=0.2, n_repeats=5)
    
    print("\n--- Random Forest (Tradicional) 90% teste ---")
    acc_90_10 = evaluate_rf(df, features, target, n_trees, max_features, max_depth, test_size=0.5, n_repeats=5)

     # Gráfico final
    plt.figure(figsize=(10, 6))
    x_labels = [f'Run {i+1}' for i in range(5)]
    plt.plot(x_labels, acc_80_20, marker='o', label='80% Treino 20% Teste')
    plt.plot(x_labels, acc_90_10, marker='s', label='50% Treino 50% Teste')
    plt.ylim(0, 1)
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácias (Random Forest)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
