# import numpy as np    

# def calculate_laplacian(adj:np.array):
#     laplacian = laplacian_matrix(adj)
#     # Calculate the maximum eigenvalue
#     eigen_values, _ = np.linalg.eig(laplacian)
#     eigen_max = max(eigen_values)
#     # Identity matrix
#     i = np.identity(adj.shape[0])

#     # Calculate L
#     scaled_laplacian = (2 * laplacian) / eigen_max - i
#     # Repeat the laplacian for number of input time steps
    # scaled_laplacian = np.array([scaled_laplacian for i in range(self.input_len)])