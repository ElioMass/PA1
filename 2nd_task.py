import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt



def similarity_matrix(pixels_np,n_samples):

    # Initialiser la matrice de similarité S avec des zéros
    S = np.zeros((n_samples, n_samples))
    # Calculer les similarités en utilisant deux boucles
    for i in range(n_samples):
        for j in range(n_samples):
            # Calcul de la similarité : - ||x_i - x_j||^2
            distance_squared = np.sum((pixels_np[i] - pixels_np[j])**2)
            S[i, j] = -distance_squared

    # Remplir la diagonale avec une valeur spécifique (par exemple, la moyenne des autres éléments)
    np.fill_diagonal(S, np.mean(S))

    print("Matrice de similarité S calculée avec dimensions :", S.shape)

    return S




# Parallel computation of R, Calcul of the matrix is seperated into calculation of part/block of the matrix
def compute_R_block(block_indices, S, A, n_samples):
    i_start, i_end = block_indices
    R_block = np.zeros((i_end - i_start, n_samples))
    
    # Formula for iterative calculation of R
    max_AS = np.max(A + S - np.eye(n_samples) * 100000, axis=1, keepdims=True)
    for i_local, i_global in enumerate(range(i_start, i_end)):
        R_block[i_local] = S[i_global] - max_AS[i_global]
    return R_block

# Parallel computation of A, Calcul of the matrix is seperated into calculation of part/block of the matrix
def compute_A_block(block_indices, R, Rp, n_samples):
    i_start, i_end = block_indices
    A_block = np.zeros((i_end - i_start, n_samples))
    
    for i_local, i_global in enumerate(range(i_start, i_end)):
        for k in range(n_samples):
            # Formula for iterative calculation of R
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

    num_processes = 1
    # num_processes = multiprocessing.cpu_count() 
    chunk_size = (n_samples + num_processes - 1) // num_processes
    blocks = [(i * chunk_size, min((i + 1) * chunk_size, n_samples)) for i in range(num_processes)]

    for iteration in range(max_iterations):
        # Update R in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            R_blocks = pool.starmap(compute_R_block, [(block, S, A, n_samples) for block in blocks])
        # Combine part to make full matrix
        R_new = np.vstack(R_blocks)

        # Pre-compute Rp (positive values of R)
        Rp = np.maximum(0, R_new)

        # Update A in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            A_blocks = pool.starmap(compute_A_block, [(block, R_new, Rp, n_samples) for block in blocks])
        # Combine part to make full matrix
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




def plot_clusters(pixels_np, representatives):
    # Find unique clusters
    unique_clusters = np.unique(representatives)
    
    for cluster_number in unique_clusters[:30]:
        # Index of points in clusters
        points_in_cluster = np.where(representatives == cluster_number)[0]
        
        print(f"Cluster {cluster_number}: {len(points_in_cluster)} points")
        
        # Plot of only 20 points
        points_to_display = points_in_cluster[:20]
        data_in_cluster = pixels_np[points_to_display]
        
        # Cluster Ploting parrt
        plt.figure(figsize=(10, 5))
        for i, index in enumerate(points_to_display):
            plt.subplot(4, 5, i + 1)  
            plt.imshow(data_in_cluster[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title(f"Pt {index}")
        
        plt.suptitle(f"Cluster {cluster_number}", fontsize=16)
        plt.tight_layout()
        plt.show()







if __name__ == "__main__":
    n_samples = 300

    # Reading of the file
    test_data = pd.read_csv('mnist_test.csv', header=0)

    # Separate  labels and pixel
    labels = test_data['label']  
    pixels = test_data.drop(columns=['label'])  

    #Let's try first with only a sample
    pixels_sampled = pixels.sample(n_samples, random_state=13)
    labels_sampled = labels.loc[pixels_sampled.index]

    # Normalizing pixels
    pixels_sampled = pixels_sampled.astype(float) / 255.0


    # Convert into numpy tab
    pixels_np = pixels_sampled.values

    # Number of points
    n_samples = pixels_np.shape[0]

    S = similarity_matrix(pixels_np,n_samples)
    print("S Matrix calculated with dimension : ", S.shape)

    R, A = parallel_affinity_propagation(S, 30, 0.95)
    print("Calcul done")

    rpz = calcul_C(R,A)

    plot_clusters(pixels_np, rpz)

    ### Obtained 89 clusters out of 500 samples

    # j'ai teste le nombre de processeur a utiliser pour peu d'echantillons ex 500 1 processeur c bcp plus rapide 
    # mais si on commence a augmenter ex 2000 echantillons plusieurs processor c bcp plus rapide