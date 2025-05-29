import pandas as pd
import numpy as np
import pprint
from typing import List, Dict, Any, Union
from collections import Counter
from utils import get_entropy, get_info_gain

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

def main():
  data_frame = pd.read_csv("./data.csv")
  features: List[str] = ['pSist','pDiast','qPA','pulso','respiracao']
  target: str = 'classe'
  final_tree: Dict[str, Any] = id3(data_frame, features, target)
  default_cl: pd.Series = data_frame[target].mode()[0]

  corrects: int = 0
  total: int = len(data_frame)
  for _, line in data_frame.iterrows():
    tested = test(line, final_tree, default_cl)
    if tested == line[target]:
      corrects+=1
      
  accuracy: float = corrects/total
  print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
  main()