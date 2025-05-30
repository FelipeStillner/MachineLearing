import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Union
from collections import Counter
from utils import get_entropy, get_info_gain
import matplotlib.pyplot as plt
import seaborn as sns

def id3(df: pd.DataFrame, atrributes: List[str], target: str) -> Union[Any, Dict[str, Any]]:
  #if all labels are equal, the label returns
  if len(df[target].unique()) == 1:
    return df[target].iloc[0]
  #if there is no more attributes, the most comman label returns
  if not atrributes:
    return Counter(df[target]).most_common(1)[0][0]
  
  #choose best attribute
  gain : Dict[str, float] = {attr: get_info_gain(df, attr, target) for attr in atrributes}
  best_attr: str = max(gain, key=gain.get)

  tree: Dict[str, Any] = {best_attr: {} }
  for value in df[best_attr].unique() :
    sub_df: pd.DataFrame = df[df[best_attr] == value]
    sub_attr:List[str] = [a for a in atrributes if a != best_attr]
    tree[best_attr][value] = id3(sub_df, sub_attr, target)

  return tree

def test( example: pd.Series, tree: Union[Any, Dict[str, Any]], default_class: Any) -> Any :
  if not isinstance(tree, dict):
    return tree
  attr: str = next(iter(tree))
  example_value: Any = example[attr]
  if example_value in tree[attr]:
    return test(example, tree[attr][example_value], default_class)
  else:
    return default_class

def evaluate(data_frame: pd.DataFrame, features: List[str], target: str, test_size: float, n_repeats: int) -> List[float]:
    accuracies = []
    for i in range(n_repeats):
        train_df, test_df = train_test_split(data_frame, test_size=test_size, train_size=1-test_size, shuffle=True, random_state=i)
        tree = id3(train_df, features, target)
        default_cl = train_df[target].mode()[0]
        
        corrects = 0
        for _, line in test_df.iterrows():
            predicted = test(line, tree, default_cl)
            if predicted == line[target]:
                corrects += 1
        acc = corrects / len(test_df)
        accuracies.append(acc)
        print(f"Repetição {i+1} ({int(test_size*100)}% teste) - Acurácia: {acc:.2%}")
    return accuracies

def main():
  data_frame = pd.read_csv("./data.csv")
  features: List[str] = ['pSist','pDiast','qPA','pulso','respiracao']
  target: str = 'classe'
  #5 fold cross-validation
  kf: KFold = KFold(n_splits=20)
  accuracies: List[float] = []

  n_repeats = 5
  
  print("\n--- Avaliando com 80% teste e 20% treino ---")
  acc_80_20 = evaluate(data_frame, features, target, test_size=0.2, n_repeats=n_repeats)
  
  print("\n--- Avaliando com 90% teste e 10% treino ---")
  acc_90_10 = evaluate(data_frame, features, target, test_size=0.1, n_repeats=n_repeats)
  
  # Gráfico final
  plt.figure(figsize=(10, 6))
  x_labels = [f'Run {i+1}' for i in range(n_repeats)]
  plt.plot(x_labels, acc_80_20, marker='o', label='80% Treino 20% Teste')
  plt.plot(x_labels, acc_90_10, marker='s', label='90% Treino 10% Teste')
  plt.ylim(0, 1)
  plt.ylabel('Acurácia')
  plt.title('Comparação de Acurácias (ID3)')
  plt.legend()
  plt.grid(True)
  plt.show()
  '''for fold,(train_index, test_index) in enumerate(kf.split(data_frame)):
    train_df: pd.DataFrame = data_frame.iloc[train_index]
    test_df: pd.DataFrame  = data_frame.iloc[test_index]

    final_tree: Dict[str, Any] = id3(train_df, features, target)
    default_cl: pd.Series = train_df[target].mode()[0]

    corrects: int = 0
    total: int = len(test_df)
    for _, line in test_df.iterrows():
      tested = test(line, final_tree, default_cl)
      if tested == line[target]:
        corrects+=1
        
    accuracy: float = corrects/total
    accuracies.append(accuracy)
    print(f"Fold: {fold+1} Accuracy: {accuracy:.2%}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\nMédia de acurácia: {mean_acc:.2%}")
    print(f"Desvio padrão: {std_acc:.2%}")
   
  # Gerar gráfico
  plt.figure(figsize=(8, 5))
  sns.barplot(x=[f'Fold {i+1}' for i in range(len(accuracies))], y=accuracies, palette='Blues_d')
  plt.axhline(mean_acc, color='red', linestyle='--', label=f'Média ({mean_acc:.2%})')
  plt.ylim(0, 1)
  plt.ylabel('Acurácia')
  plt.title('Acurácia por Fold - ID3 com 20-Fold')
  plt.legend()
  plt.show()''' 


if __name__ == "__main__":
  main()