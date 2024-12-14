from multiprocessing import Pool
import multiprocessing
import math
from collections import defaultdict

# Load the graph and build the adjacency list
def load_graph_as_adjacency_list(file_path):
    adjacency_list = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Ignore the 4 first lines with #
                continue
            from_vertex, to_vertex = map(int, line.split())  
            adjacency_list[from_vertex].append(to_vertex) # create connection between from vertex and to vertex
    return adjacency_list

# The reverse_adjacency_list_N_u correspond to the N^-(u) list that we need in the calculation of the pagerank
def build_reverse_adjacency_list_N_u(adjacency_list):
    reverse_adjacency_list = defaultdict(list)
    for from_vertex, neighbors in adjacency_list.items():
        for to_vertex in neighbors:
            reverse_adjacency_list[to_vertex].append(from_vertex)
    return reverse_adjacency_list

# page rank calculation of a subset of vertexs
def compute_page_rank_for_vertexs_local(vertexs, reverse_adj_list_local, current_page_rank_list, d, out_degrees, num_vertexs):
    new_page_ranks = {}
    for u in vertexs:
        # for loop on the N^-(u)
        pr_sum = sum(current_page_rank_list[v] / out_degrees[v] for v in reverse_adj_list_local[u])
        # Calcul of page rank of vertex u 
        new_page_ranks[u] = (1 - d) / num_vertexs + d * pr_sum
    return new_page_ranks


if __name__ == "__main__":
    # load the graph
    file_path = "web-BerkStan.txt"
    adjacency_list = load_graph_as_adjacency_list(file_path)
    reverse_adjacency_list = build_reverse_adjacency_list_N_u(adjacency_list)

    # Parameters
    num_vertexs = len(adjacency_list)
    d = 0.85
    max_iterations = 100
    tolerance = 1e-6
    num_processes = 2  # multiprocessing.cpu_count() 

    # Initialisation of page rank list with the value 1/V
    page_rank_list = {vertex: 1 / num_vertexs for vertex in adjacency_list}
    out_degrees = {vertex: len(neighbors) for vertex, neighbors in adjacency_list.items()}

    # # Calcul of all PageRank
    for iteration in range(max_iterations):
        # divide vertexs into chunks
        vertexs = list(adjacency_list.keys())
        chunk_size = math.ceil(len(vertexs) / num_processes)
        chunks = [vertexs[i:i + chunk_size] for i in range(0, len(vertexs), chunk_size)]

        # Parallel calculation
        with Pool(num_processes) as pool:
            results = pool.starmap(
                compute_page_rank_for_vertexs_local,
                [
                    (chunk, reverse_adjacency_list, page_rank_list, d, out_degrees, num_vertexs)
                    for chunk in chunks
                ]
            )

        # fuse the different part of the calculated page rank list
        new_page_rank_list = {}
        for result in results:
            new_page_rank_list.update(result)

        # check if the change is significantly low
        diff = sum(abs(new_page_rank_list[u] - page_rank_list[u]) for u in page_rank_list)
        if diff < tolerance:
            print(f"convergence reached at iteration {iteration + 1}")
            break

        # Update page rank
        page_rank_list = new_page_rank_list
        print(f"Iteration {iteration} done")

    # Printing of some vertexs to see if worked
    print("PageRanks of 10 first vertexs :")
    for vertex, pr in list(page_rank_list.items())[:10]:
        print(f"vertex {vertex} : {pr}")

