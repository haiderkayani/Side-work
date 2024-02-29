import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

df = pd.read_excel("Data.xlsx")

te = TransactionEncoder()
te_ary = te.fit(df.values).transform(df.values)
encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

min_support_threshold = 0.2
frequent_itemsets_apriori = apriori(encoded_df, min_support=min_support_threshold, use_colnames=True)
frequent_itemsets_fpgrowth = fpgrowth(encoded_df, min_support=min_support_threshold, use_colnames=True)

min_confidence_threshold = 0.7

rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence_threshold)
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence_threshold)

print("Apriori Frequent Itemsets:")
print(frequent_itemsets_apriori)
print("\nFP-Growth Frequent Itemsets:")
print(frequent_itemsets_fpgrowth)

print("\nApriori Association Rules:")
print(rules_apriori)
print("\nFP-Growth Association Rules:")
print(rules_fpgrowth)
