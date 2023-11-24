# Visualize splits
import plotly.graph_objects as go
from collections import Counter
import numpy as np

def get_label_freqs(files, label_representations, file_list):
    label_freqs = Counter()
    for i in range(len(file_list)):
        if file_list[i] in files:
            label_freqs.update(label_representations[i])
    return label_freqs

train_freqs = get_label_freqs(train_files, label_representations, file_list)
val_freqs = get_label_freqs(val_files, label_representations, file_list)
test_freqs = get_label_freqs(test_files, label_representations, file_list)

labels, train_counts, val_counts, test_counts = [], [], [], []
for label in set(train_freqs) | set(val_freqs) | set(test_freqs):
    labels.append(label)
    train_counts.append(train_freqs[label])
    val_counts.append(val_freqs[label])
    test_counts.append(test_freqs[label])

train_log = np.log(np.array(train_counts) + 1)
val_log = np.log(np.array(val_counts) + 1)
test_log = np.log(np.array(test_counts) + 1)

# Plot using plotly
trace = go.Scatter3d(
    x=train_log,
    y=val_log,
    z=test_log,
    mode='markers',
    marker=dict(
        size=5,
        opacity=0.8
    ),
    text=labels
)

layout = go.Layout(
    title='3D Scatter plot of Graph Frequencies (Log scale)',
    scene=dict(
        xaxis_title='Log(Train Frequency)',
        yaxis_title='Log(Validation Frequency)',
        zaxis_title='Log(Test Frequency)',
        annotations=[
            dict(
                x=max(train_log),
                y=max(val_log),
                z=max(test_log),
                text=f"Train Files: {len(train_files)}",
                showarrow=False
            ),
            dict(
                x=max(train_log),
                y=max(val_log),
                z=0,
                text=f"Validation Files: {len(val_files)}",
                showarrow=False
            ),
            dict(
                x=max(train_log),
                y=0,
                z=max(test_log),
                text=f"Test Files: {len(test_files)}",
                showarrow=False
            )
        ]
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()
