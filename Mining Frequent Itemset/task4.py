import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

min_support_threshold = 0.2
frequent_itemsets = fpgrowth(encoded_df, min_support=min_support_threshold, use_colnames=True)

min_confidence_threshold = 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_threshold)

print(rules)
