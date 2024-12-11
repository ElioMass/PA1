import itertools
import multiprocessing

# Writing the srflp Instance
srflp = [
    [10], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
    [0, 30, 17, 11, 24, 25, 24, 17, 16, 22],
    [0, 0, 21, 23, 26, 24, 27, 19, 11, 32],
    [0, 0, 0, 24, 18, 23, 31, 36, 28, 19],
    [0, 0, 0, 0, 19, 18, 33, 25, 20, 28],
    [0, 0, 0, 0, 0, 15, 37, 27, 17, 16],
    [0, 0, 0, 0, 0, 0, 27, 23, 29, 24],
    [0, 0, 0, 0, 0, 0, 0, 27, 31, 24],
    [0, 0, 0, 0, 0, 0, 0, 0, 14, 18],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 24],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# Exctrating the value from the srflp instance 
n = srflp[0][0]
widths = srflp[1]
weights = [row[:] for row in srflp[2:]]

#I separeted the calculation of the cost in 2 functions
# First the calculation of the sum of the facility widths
def distance_facilities(p, i, j):
    dist = (widths[p[i]] + widths[p[j]]) / 2
    for k in range(min(i,j) + 1,max(i,j)):
        dist += widths[p[k]]
    return dist

# Then calculation of the total cost of a permutation 
def srflp_permutation_cost(p):
    cost = 0
    for i in range(n):
        for j in range(i + 1, n):
            fac1, fac2 = p[i], p[j]
            weight = weights[fac1][fac2] if fac1 < fac2 else weights[fac2][fac1]
            cost += weight * distance_facilities(p, i, j)
    return cost


# Fonction that tracks the best cost and so the best permutation of a list of permutation 
def evaluate_permutations(perm_subset):
    min_cost = float('inf')
    best_perm = None
    for perm in perm_subset:
        cost = srflp_permutation_cost(perm)
        if cost < min_cost:
            min_cost = cost
            best_perm = perm
    return min_cost, best_perm

# Generate the list of all permutation and then divide it in subset
def generate_permutation_subsets(num_chunks):
    permutations = list(itertools.permutations(range(n)))
    chunk_size = len(permutations) // num_chunks
    return [permutations[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()  
    permutation_subsets = generate_permutation_subsets(num_processes)
    
    # Applying the evaluating function to the subsets of permutations
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_permutations, permutation_subsets)

    # Out of the best cost found in each funtion we tak the minimum 
    best_cost, best_permutation = min(results, key=lambda x: x[0])

    print("Optimal permutation:", best_permutation)
    print("Minimum cost:", best_cost)
