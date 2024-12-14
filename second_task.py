import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt


#Calculation of S 
def similarity_matrix(pixels_np,n_samples):

    # Initialize with zeros
    S = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            #Formula of the squared distance 
            distance_squared = np.sum((pixels_np[i] - pixels_np[j])**2)
            S[i, j] = -distance_squared

    # Diagonal filled with mean value according to the documentation 
    np.fill_diagonal(S, np.mean(S))

    return S




# Parallel computation of R
def compute_R_block(block_indices, S, A, n_samples):
    i_start, i_end = block_indices
    R_block = np.zeros((i_end - i_start, n_samples))
    
    max_AS = np.max(A + S - np.eye(n_samples) * 100000, axis=1, keepdims=True)
    for i_local, i_global in enumerate(range(i_start, i_end)):
        R_block[i_local] = S[i_global] - max_AS[i_global]
    return R_block

# Parallel computation of A
def compute_A_block(block_indices, R, Rp, n_samples):
    i_start, i_end = block_indices
    A_block = np.zeros((i_end - i_start, n_samples))
    
    for i_local, i_global in enumerate(range(i_start, i_end)):
        for k in range(n_samples):
            if i_global != k:
                sum_term = Rp[:, k].sum() - Rp[i_global, k]
                A_block[i_local, k] = min(0, R[k, k] + sum_term)
            else:
                A_block[i_local, k] = Rp[:, k].sum() - Rp[k, k]
    return A_block


# Main parallel affinity propagation function
def parallel_affinity_propagation(S, max_iterations, damping_factor):
    n_samples = S.shape[0]
    R = np.zeros((n_samples, n_samples))
    A = np.zeros((n_samples, n_samples))

    #part to choose to devide the task in different processes or not
    num_processes = 1
    # num_processes = multiprocessing.cpu_count() 
    chunk_size = (n_samples + num_processes - 1) // num_processes
    blocks = [(i * chunk_size, min((i + 1) * chunk_size, n_samples)) for i in range(num_processes)]

    for iteration in range(max_iterations):
        # Update R in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            R_blocks = pool.starmap(compute_R_block, [(block, S, A, n_samples) for block in blocks])
        R_new = np.vstack(R_blocks)

        # Pre-compute Rp (positive values of R)
        Rp = np.maximum(0, R_new)

        # Update A in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            A_blocks = pool.starmap(compute_A_block, [(block, R_new, Rp, n_samples) for block in blocks])
        A_new = np.vstack(A_blocks)

        # Apply damping factor
        R = damping_factor * R + (1 - damping_factor) * R_new
        A = damping_factor * A + (1 - damping_factor) * A_new

        print(f"Iteration {iteration + 1}/{max_iterations} completed.")

    return R, A



def calcul_C(R,A) :
    # Criterion Matrix
    C = R + A

    #Assigning of clusters
    representatives = np.argmax(C, axis=1)  
    unique_representatives = np.unique(representatives)
    clusters = {rep: [] for rep in unique_representatives}  
    for i, rep in enumerate(representatives):
        clusters[rep].append(i)

    print("Number of indetified clusters :", len(unique_representatives))
    for rep, points in clusters.items():
        print(f"Number of point in Cluster {rep} :  {len(points)} ")
    
    return representatives




def plot_clusters(pixels_np, representatives,cluster_number):
    
    points_in_cluster = np.where(representatives == cluster_number)[0]
        
    print(f"Cluster {cluster_number}: {len(points_in_cluster)} points")
        
    # Plot only 20 points per clusters
    points_to_display = points_in_cluster[:20]
    data_in_cluster = pixels_np[points_to_display]
        
    # Creating of a plot for the cluster
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(points_to_display):
        plt.subplot(6, 5, i + 1) 
        plt.imshow(data_in_cluster[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(f"Pt {index}")
        
    plt.suptitle(f"Cluster {cluster_number}", fontsize=16)
    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    n_samples = 1000

    # Reading of the file
    test_data = pd.read_csv('mnist_test.csv', header=0)

    # Separate  labels and pixel
    labels = test_data['label']  
    pixels = test_data.drop(columns=['label'])  

    # We will work with only a sample
    pixels_sampled = pixels.sample(n_samples, random_state=13)
    labels_sampled = labels.loc[pixels_sampled.index]

    # Normalizing pixels
    pixels_sampled = pixels_sampled.astype(float) / 255.0


    # Convert into numpy tab for calculation
    pixels_np = pixels_sampled.values

    # Number of points
    n_samples = pixels_np.shape[0]

    # Calculation of S 
    S = similarity_matrix(pixels_np,n_samples)
    print("S Matrix calculated with dimension : ", S.shape)

    # Calculation of R and A 
    R, A = parallel_affinity_propagation(S, 400, 0.95)
    print("Calcul done")

    # Calculation of C
    rpz = calcul_C(R,A)

    plot_clusters(pixels_np, rpz, 4)

    ### Obtained 161 clusters out of 1000 samples

     
