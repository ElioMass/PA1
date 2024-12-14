from collections import defaultdict

# Load the graph and build the adjacency list
def load_graph_as_adjacency_list(file_path):
    adjacency_list = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Ignore the 4 first lines with #
                continue
            from_vertex, to_vertex = map(int, line.split())
            adjacency_list[from_vertex].append(to_vertex)  # create connection between from vertex and to vertex
    return adjacency_list

# load the graph
file_path = "web-BerkStan.txt"
adjacency_list = load_graph_as_adjacency_list(file_path)

# Optional part for me to check if the loading of the graph and adjacencylist correctly worked
# print("Example of adjacency_list :")
# for vertex, neighbors in list(adjacency_list.items())[:5]:  
#     print(f"vertex {vertex} -> {neighbors}")

# vertices and edges
num_edges = sum(len(neighbors) for neighbors in adjacency_list.values())
num_vertices = len(adjacency_list)
print(f"number of vertices : {num_vertices}")
print(f"Number of edges : {num_edges}")


# The reverse_adjacency_list_N_u correspond to the N^-(u) list that we need in the calculation of the pagerank
def build_reverse_adjacency_list_N_u(adjacency_list):
    reverse_adjacency_list = defaultdict(list)
    for from_vertex, neighbors in adjacency_list.items():
        for to_vertex in neighbors:
            reverse_adjacency_list[to_vertex].append(from_vertex)
    return reverse_adjacency_list

reverse_adjacency_list = build_reverse_adjacency_list_N_u(adjacency_list)

#page rank calculation of a vertex u
def Page_rank_calc(u, reverse_adj_list, adj_list, page_rank_list, d):
    V = len(adj_list) 
    pr_sum = 0

    # for loop on the N^-(u)
    for v in reverse_adj_list[u]:
        pr_sum += page_rank_list[v] / len(adj_list[v])  

    # Calcul of page rank of vertex u 
    page_rank_u = (1-d)/V + d * pr_sum
    return page_rank_u


# Initialisation of page rank list with the value 1/V
page_rank_list = {vertex: 1 / num_vertices for vertex in adjacency_list}

# Parameters
d = 0.85
max_iterations = 100
tolerance = 1e-6

# Calcul of PageRank
for iteration in range(max_iterations):
    new_page_rank_list = {}
    for u in adjacency_list:
        new_page_rank_list[u] = Page_rank_calc(u, reverse_adjacency_list, adjacency_list, page_rank_list, d)


    # check if the change is significantly low
    diff = sum(abs(new_page_rank_list[u] - page_rank_list[u]) for u in page_rank_list)
    if diff < tolerance:
        print(f"convergence reached at iteration {iteration + 1}")
        break

    # Update page rank
    page_rank_list = new_page_rank_list
    print(f"iteration {iteration} done")

# Printing of some vertices to see if worked
print("PageRanks of 10 first vertices :")
for vertex, pr in list(page_rank_list.items())[:10]:
    print(f"vertex {vertex} : {pr}")




