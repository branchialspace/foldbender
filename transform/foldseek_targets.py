# Foldseek task multitarget regression clusters + fident

import pandas as pd

# Create file_clusters with foldseek command: !foldseek easy-cluster /content/41k_go_filtered/ res clust
# Create file_scores with foldseek command: !foldseek easy-search /content/41k_go_filtered/ /content/41k_go_filtered/ aln tmp --format-output "query,target,fident,rmsd,evalue"

def foldseek_scored_clusters(file_clusters, file_scores, file_output):
    # Read the clusters file
    df_clusters = pd.read_csv(file_clusters, sep='\t', header=None, names=['representative', 'member'])

    # Filter representatives present 3 or more times
    filtered_reps = df_clusters['representative'].value_counts()
    filtered_reps = filtered_reps[filtered_reps >= 3].index.tolist()

    # Create a list of all unique members in the clusters file
    all_members = df_clusters['member'].unique()

    # Read the similarity scores file
    df_scores = pd.read_csv(file_scores, sep='\t', header=None, names=['representative', 'member', 'score'], usecols=[0, 1, 2])

    # Convert scores to float32
    df_scores['score'] = df_scores['score'].astype('float32')

    # Filter scores by representatives in the filtered list and members from the clusters file
    df_scores_filtered = df_scores[df_scores['representative'].isin(filtered_reps) & df_scores['member'].isin(all_members)]

    # Group by member and get the top 3 scores for each member
    top_scores = df_scores_filtered.groupby('member', group_keys=False).apply(lambda x: x.nlargest(3, 'score')).reset_index(drop=True)

    # Count scores for each member
    member_score_counts = top_scores['member'].value_counts()
    members_with_2_scores = member_score_counts[member_score_counts == 2].count()
    members_with_1_score = member_score_counts[member_score_counts == 1].count()
    members_with_0_scores = len(all_members) - member_score_counts.count()

    print(f"Number of members with 2 scores: {members_with_2_scores}")
    print(f"Number of members with 1 score: {members_with_1_score}")
    print(f"Number of members with 0 scores: {members_with_0_scores}")

    # Write to the output TSV file
    top_scores.to_csv(file_output, sep='\t', index=False, header=False)

if __name__ == "__main__":

    file_clusters = '/content/drive/MyDrive/protein-DATA/res_cluster.tsv'
    file_scores = '/content/drive/MyDrive/protein-DATA/aln2.tsv'
    file_output = '/content/foldseek_labels.tsv'
  
    foldseek_scored_clusters(file_clusters, file_scores, file_output)
