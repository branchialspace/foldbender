import pandas as pd

df = pd.read_csv('your_file.tsv', sep='\t')

# Ensure each EntryID has at least one row for each of three different aspects
valid_entry_ids = df.groupby('EntryID').filter(lambda x: x['aspect'].nunique() == 3)['EntryID'].unique()
df = df[df['EntryID'].isin(valid_entry_ids)]

# Count the occurrence of each term and filter out those with less than 50 occurrences
term_counts = df['term'].value_counts()
terms_with_at_least_50 = term_counts[term_counts >= 50].index
df = df[df['term'].isin(terms_with_at_least_50)]

print(f"Total number of unique EntryID values: {df['EntryID'].nunique()}")
print(f"Total number of unique term values: {df['term'].nunique()}")

df.to_csv('modified_file.tsv', sep='\t', index=False)
