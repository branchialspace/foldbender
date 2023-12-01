import pandas as pd

def bin_and_count_terms(file_path, count_threshold):

    df = pd.read_csv(file_path, sep='\t')

    # Count occurrences of each unique 'term'
    term_counts = df['term'].value_counts()

    # Count unique terms above the specified threshold
    terms_above_threshold = term_counts[term_counts > count_threshold].count()

    # Define bins
    bins = list(range(0, 101, 25)) + list(range(200, term_counts.max() + 100, 100))
    bin_labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]

    # Bin the term counts
    binned_term_counts = pd.cut(term_counts, bins=bins, labels=bin_labels, right=False)

    # Count the number of 'term' values in each bin
    binned_term_counts = binned_term_counts.value_counts().sort_index()

    return binned_term_counts.to_dict(), terms_above_threshold

file_path = '/content/drive/MyDrive/filtered_file.tsv'
count_threshold = 49
binned_counts, above_threshold = bin_and_count_terms(file_path, count_threshold)

print("Binned term counts:", binned_counts)
print("Number of unique 'term' values with count above", count_threshold, ":", above_threshold)
