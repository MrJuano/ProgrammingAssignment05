# Import the packages 
import numpy as np
#load the transactions dataset 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Loading the data
def load_dataset(path_to_data):
    transactions = []
    with open(path_to_data, 'r') as fid:
        for lines in fid:
            transaction = lines.strip().split(',')
            transactions.append(transaction)
    return transactions

path_to_data = "transactions_data.txt"  
dataset = load_dataset(path_to_data)
dataset

# Transform the data to a format suitable for the apriori function
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the apriori algorithm
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True) #YOUR CODE  
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate the association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:")
print(rules)

# Transform the data to a format suitable for the apriori function
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the apriori algorithm
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True) #YOUR CODE  
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate the association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:")
print(rules)