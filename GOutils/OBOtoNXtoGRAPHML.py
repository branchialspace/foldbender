# Gene Ontology OBO > NetworkX > GraphML
import networkx as nx
import obonet

# Translate the Gene Ontology to NetworkX Directed Acyclic Graph/ MultiDiGraph
GO_graph = obonet.read_obo(GO_TERMS)

# Initialize a new graph
new_GO_graph = nx.DiGraph()

# Copy nodes and data, skip data of 'type' and convert lists to strings
for node, data in GO_graph.nodes(data=True):
    new_data = {}
    for key, value in data.items():
        if not isinstance(value, type):
            if isinstance(value, list):
                new_data[key] = str(value)
            else:
                new_data[key] = value
    new_GO_graph.add_node(node, **new_data)

# Copy edges and data, skip data of 'type' and convert lists to strings
for u, v, data in GO_graph.edges(data=True):
    new_data = {}
    for key, value in data.items():
        if not isinstance(value, type):
            if isinstance(value, list):
                new_data[key] = str(value)
            else:
                new_data[key] = value
    new_GO_graph.add_edge(u, v, **new_data)

# Write to GraphML
nx.write_graphml(new_GO_graph, "gene_ontology.graphml")
