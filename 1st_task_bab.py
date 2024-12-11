import itertools
import math
from multiprocessing import Pool, Manager

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

# Branch-and-bound function for a specific branch
def branch_and_bound_partial(start_perm, best_cost_manager):
    best_cost_local = best_cost_manager.value
    best_permutation_local = None

    def branch_and_bound(permutation, used, current_cost):
        nonlocal best_cost_local, best_permutation_local

        # If the permutation is complete, we can evaluate its cost
        if len(permutation) == n:
            if current_cost < best_cost_local:
                best_cost_local = current_cost
                best_permutation_local = permutation[:]
                best_cost_manager.value = best_cost_local
            return

        # Branching for each possible facility
        for i in range(n):
            if not used[i]:  # If the facility hasn't been used yet
                new_cost = current_cost
                for j in range(len(permutation)):
                    fac1, fac2 = permutation[j], i
                    weight = weights[fac1][fac2] if fac1 < fac2 else weights[fac2][fac1]
                    new_cost += weight * distance_facilities(permutation + [i], j, len(permutation))

                # Bounding
                if new_cost >= best_cost_local:
                    continue

                # Mark this facility as used and explore further
                used[i] = True
                permutation.append(i)

                branch_and_bound(permutation, used, new_cost)

                # Backtracking
                permutation.pop()
                used[i] = False

    # Start the search for this branch
    branch_and_bound(start_perm, [i in start_perm for i in range(n)], 0)
    return best_permutation_local, best_cost_local

# Initialization for parallel search
def parallel_branch_and_bound():
    with Manager() as manager:
        best_cost_manager = manager.Value('d', float('inf'))
        best_permutation = None

        # Define starting branches for each processs
        initial_perms = [[i] for i in range(n)]
        
        with Pool() as pool:
            results = pool.starmap(branch_and_bound_partial, [(perm, best_cost_manager) for perm in initial_perms])

        # Get the best result among branches
        for permutation, cost in results:
            if cost < best_cost_manager.value:
                best_permutation = permutation
                best_cost_manager.value = cost

        return best_permutation, best_cost_manager.value

# Execute the function in the main block
if __name__ == '__main__':
    best_permutation, best_cost = parallel_branch_and_bound()
    print("Optimal permutation:", best_permutation)
    print("Minimum cost:", best_cost)
