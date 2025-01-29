import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
import datetime
import matplotlib.pyplot as plt

from PIL import Image
from ImageModel import SimpleNNBinary


# Set Seed for Reproducibility

def set_seed(seed):
    """
    Set various random seeds to ensure reproducible results.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the global seed
set_seed(42)


# Initialization & Paths

f_star_test = False
time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(time, flush=True)

# These paths should be replaced
SAVE_PATH = ''     # Directory to save results
MODEL_PATH = ''    # Path to the model weights file
train_dir = ''     # Training images directory

# Create the save path if it does not exist (example usage)
os.mkdir(SAVE_PATH)


# Load Model & Model Weights

model = SimpleNNBinary()  # Generic example model; replace if needed
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Access weights for further computation
weights_b = model.fc1.weight.data  # For example, the first layer's weights
weights_a = model.fc2.weight.data  # For example, the second layer's weights

# Combine weights into a single parameter vector for gradient-based computations
theta = [weights_a, weights_b]
print(len(theta[1]), flush=True)

# Flatten weights for integration-based computation (example: 10 neurons)
a_flat = weights_a.view(-1)    # Flatten 'a' to shape (10,)
b_flat = weights_b.view(-1)    # Flatten 'b' to shape (1000,)
theta_f_star_numerical = np.concatenate([
    np.concatenate([a_flat[i:i+1], b_flat[i*100:(i+1)*100]]) 
    for i in range(10)
])
print(theta_f_star_numerical, theta_f_star_numerical.shape, flush=True)

weights_a, weights_b = weights_a.numpy(), weights_b.numpy()
print(weights_a.shape, weights_b.shape, flush=True)


#  Neural Network Helper Functions

def f1(theta, x):
    """
    Compute f(θ, x) = sigmoid(a ⋅ ReLU(b x))

    Args:
        theta (Tensor): Shape (1010,). Concatenated vector of [a_0, b_0, ..., a_9, b_9].
        x (Tensor): Shape (100,). Input data (flattened image, etc.).

    Returns:
        Scalar (Tensor): Output of the network (a single float).
    """
    num_neurons = 10
    neuron_params = 1 + 100  # Each neuron has 1 a_i and 100 b_i

    a = torch.zeros(1, num_neurons, dtype=theta.dtype)
    b = torch.zeros(num_neurons, 100, dtype=theta.dtype)

    for i in range(num_neurons):
        start_idx = i * neuron_params
        end_idx = start_idx + neuron_params

        a_i = theta[start_idx]
        b_i = theta[start_idx + 1:end_idx]

        a[0, i] = a_i
        b[i, :] = b_i

    z = b @ x              # Shape: (10,)
    h = torch.relu(z)      # ReLU activation
    s = a @ h              # Inner product with 'a'
    y = torch.sigmoid(s)   # Sigmoid output

    return y.squeeze()

def compute_f_star(theta, x, N=5000):
    """
    Compute f*(θ, x) = ∫₀¹ ∂/∂θ f(tθ, x) dt using numerical integration.

    Args:
        theta (Tensor): Shape (1010,). Parameter vector for the network.
        x (Tensor): Shape (100,). Input vector.
        N (int): Number of steps for numerical integration (default: 5000).

    Returns:
        f_star (Tensor): Shape (1010,). Computed integral approximation.
    """
    # Prepare tensors
    theta = theta.clone().detach().float().requires_grad_(True)
    x = x.clone().detach().float()
    f_star = torch.zeros_like(theta)

    # Discretize t from 0 to 1 (excluding 0 to avoid zero grads)
    t_values = torch.linspace(0, 1, N + 1, dtype=torch.float32)[1:]

    for t in t_values:
        t = t.item()
        theta_t = t * theta
        y_t = f1(theta_t, x)

        # Compute gradient of y_t w.r.t. theta
        grad_theta = torch.autograd.grad(outputs=y_t, 
                                         inputs=theta,
                                         retain_graph=False,
                                         create_graph=False)[0]
        # Accumulate trapezoidal or Riemann sum
        f_star += grad_theta / N

    return f_star


# Example: Test f* Functionality (Optional)

if f_star_test:
    x = torch.randn(100)  # Example input vector
    f_star_numerical = compute_f_star(torch.from_numpy(theta_f_star_numerical), x, N=5000)
    print("Computed f*(theta, x):", f_star_numerical, flush=True)


# Additional Mathematical Helper Functions

def psi(b, x):
    """
    Calculate ReLU(b ⋅ x).

    Args:
        b (np.array): Weights for the neuron.
        x (np.array): Input vector.

    Returns:
        float: ReLU(b ⋅ x).
    """
    return max(np.dot(b, x), 0)

def phi(x):
    """
    Sigmoid activation: 1 / (1 + exp(-x)).
    """
    return 1/(1 + np.exp(-x))

def phi_star(alpha, a, b, x=np.random.rand(100)):
    """
    Custom function to calculate a contribution for each neuron based on:
        phi(a * ReLU(bx)) - phi(0) / (alpha * a * ReLU(bx)) * [ReLU(bx), a * x].
    
    Args:
        alpha (float): Scaling parameter.
        a (float): Weight for the neuron (scalar).
        b (np.array): Weights for the neuron (vector).
        x (np.array): Input data.

    Returns:
        np.array: Computed contribution of shape (len(x)+1,).
    """
    if psi(b, x) == 0:
        return np.zeros(len(x) + 1)

    # Scale factor
    factor = (phi(a * psi(b, x)) - phi(0)) / (alpha * a * psi(b, x))

    # Build final vector
    value = a * x
    if psi(b, x) <= 0:
        value = 0

    output_vector = np.concatenate([np.array([psi(b, x)]), value])
    return factor * output_vector

def f_star_row(alpha, theta, x=np.random.rand(100)):
    """
    Combines phi_star() for each neuron into a single row (vector).
    
    Args:
        alpha (float): Scaling parameter.
        theta (list): [weights_a, weights_b], 
                      where weights_a is shape (1, num_neurons),
                            weights_b is shape (num_neurons, 100).
        x (np.array): Input data.

    Returns:
        list: Combined contributions from all neurons.
    """
    # theta[0] -> weights_a shape: (1, 10)
    # theta[1] -> weights_b shape: (10, 100)
    # We assume 10 neurons here for illustration
    f_star_matrix = []
    for i in range(len(theta[0][0])):
        a = theta[0][0][i]
        b = theta[1][i]
        f_star_matrix += phi_star(alpha, a, b, x).tolist()
    return f_star_matrix


#  Loading & Resizing Images from Folders

def load_resize_images_with_paths(parent_directory, target_size=(10, 10)):
    """
    Load images from a parent directory with known class subdirectories
    (e.g. '1', '2'), resize, and return flattened arrays with paths.

    Args:
        parent_directory (str): Parent directory containing class folders.
        target_size (tuple): Desired (width, height).

    Returns:
        (np.array, list): (Flattened image arrays, list of file paths)
    """
    images = []
    paths = []
    # You can modify the subdirectories list to suit your dataset
    subdirectories = ['1', '2']  
    for subdirectory in subdirectories:
        dir_path = os.path.join(parent_directory, subdirectory)
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_path, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_resized = img.resize(target_size, resample=Image.Resampling.LANCZOS)
                img_array = np.array(img_resized).flatten()
                images.append(img_array)
                paths.append(img_path)
    return np.array(images), paths

# Load train images
train_images, train_paths = load_resize_images_with_paths(train_dir)

# Create dataframes for image paths
train_df = pd.DataFrame({'Index': range(len(train_paths)), 'Path': train_paths})

# Save dataframes
train_df.to_csv(f'{SAVE_PATH}/train_df.csv', index=False)


# Visualize First Train Image for Verification

plt.imshow(train_images[0].reshape(10, 10), cmap='gray')
plt.title("First Train Image (Resized to 10x10)")
plt.axis('off')
plt.show()
plt.close()


#  Compute Non-Linear Leverage Scores (Example)

f_star_matrix_ls = []
X = train_images
alpha = 2
m = len(theta[1])  # e.g., number of neurons times weight dimension
image_dict = {}

method_difference_1 = []
method_difference_2 = []

# Loop over each example in training images
for x in range(len(X)):
    if x % 100 == 0:
        print(f"Processing image index: {x}", flush=True)

    # Row from the custom function
    row_to_append = f_star_row(alpha, theta, X[x])
    
    # Row from numerical integration
    f_star_num = compute_f_star(
        torch.from_numpy(theta_f_star_numerical).float(),
        torch.from_numpy(X[x]).float(),
        N=5000
    ).tolist()

    # Compare the two methods
    method_difference_1.append(
        np.linalg.norm(np.array(row_to_append) - np.array(f_star_num))
    )

    # Example condition: if row_to_append is not all zeros
    # (this is just a placeholder logic for demonstration)
    if all(int(element) == 0 for element in row_to_append) != 0:
        method_difference_2.append(
            np.linalg.norm(np.array(row_to_append) - np.array(f_star_num))
        )
        row_to_append.append(m * phi(0))  # Append an extra term if needed
        f_star_matrix_ls.append(row_to_append)
        image_dict[len(image_dict)] = train_df.iloc[x]['Path']

# Create a dataframe for selected images
image_df = pd.DataFrame(list(image_dict.items()), columns=['Index', 'Path'])
image_df.to_csv(f"{SAVE_PATH}/image_df.csv", index=False)

f_star_matrix_ls = np.array(f_star_matrix_ls)
print(f_star_matrix_ls.shape, flush=True)


#  Calculate & Plot Method Differences

plt.plot(method_difference_1)
plt.title("Method Difference 1")
plt.savefig(f'{SAVE_PATH}/Method_Difference_1.png')
plt.close()

plt.plot(method_difference_2)
plt.title("Method Difference 2")
plt.savefig(f'{SAVE_PATH}/Method_Difference_2.png')
plt.close()

print("Method Difference 1:", method_difference_1, flush=True)
print("Method Difference 2:", method_difference_2, flush=True)
print('MD1 Average:', sum(method_difference_1)/len(method_difference_1), flush=True)
print('MD2 Average:', sum(method_difference_2)/len(method_difference_2), flush=True)


#  Compute Statistical Leverage Scores (NLLS Example)

nlls = np.zeros(len(f_star_matrix_ls))
f_star_matrix_ls_XTX = np.dot(np.transpose(f_star_matrix_ls), f_star_matrix_ls)

rank = np.linalg.matrix_rank(f_star_matrix_ls_XTX)
print("Rank of f_star_matrix_ls_XTX:", rank, flush=True)
print("Shape of f_star_matrix_ls_XTX:", f_star_matrix_ls_XTX.shape, flush=True)

# Identify zero rows and columns (debugging)
zero_rows = np.where(~f_star_matrix_ls.any(axis=1))[0]
zero_cols = np.where(~f_star_matrix_ls.any(axis=0))[0]
print("Zero rows:", zero_rows, flush=True)
print("Zero columns:", zero_cols, flush=True)

condition_number = np.linalg.cond(f_star_matrix_ls_XTX)
print("Condition number of f_star_matrix_ls_XTX:", condition_number, flush=True)

# Compute pseudoinverse
f_star_matrix_ls_XTX_inv = np.linalg.pinv(f_star_matrix_ls_XTX)

# Calculate leverage scores for each row
for index in range(len(nlls)):
    x_vec = f_star_matrix_ls[index]
    nlls[index] = np.dot(np.dot(np.transpose(x_vec), f_star_matrix_ls_XTX_inv), x_vec)

# Normalize to get probability distribution
prob = nlls / np.sum(nlls)
print('Probability:', prob, flush=True)
print('Sum of Probabilities:', np.sum(prob), flush=True)

# Plot leverage scores
plt.plot(prob)
plt.title("Normalized Leverage Scores")
plt.savefig(f'{SAVE_PATH}/NLLS.png')
plt.close()


# Helper Functions to Identify Top / Bottom M Probability Indices

def top_m_indices_and_values(lst, m):
    """
    Return indices and values of the top m elements in a list.
    """
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    top_m_indices = sorted_indices[:m]
    top_m_values = [lst[i] for i in top_m_indices]
    return top_m_indices, top_m_values

def bottom_m_indices_and_values(lst, m):
    """
    Return indices and values of the bottom m elements in a list.
    """
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])
    bottom_m_indices = sorted_indices[:m]
    bottom_m_values = [lst[i] for i in bottom_m_indices]
    return bottom_m_indices, bottom_m_values

# Example usage of the above helper functions
mm = 10000  # Number of images to retrieve (replace with actual count as needed)
top_m_indices, top_m_values = top_m_indices_and_values(prob, mm)
bottom_m_indices, bottom_m_values = bottom_m_indices_and_values(prob, mm)


# Save Results in CSV for Top & Bottom Scoring Rows

all_indices_descending, all_values_descending = top_m_indices_and_values(prob, len(prob))

def print_paths_for_indices(df, indices):
    """
    Given a dataframe with 'Index' and 'Path' columns, and a list of indices,
    return a dictionary mapping index -> path.
    """
    results = {}
    for idx in indices:
        path = df.loc[df['Index'] == idx, 'Path'].iloc[0]
        results[idx] = path
    return results

top_result_dict = print_paths_for_indices(image_df, top_m_indices)
top_result_df = pd.DataFrame(list(top_result_dict.items()), columns=['Index', 'Path'])
top_result_df.to_csv(f'{SAVE_PATH}/top_results.csv', index=False)

bottom_result_dict = print_paths_for_indices(image_df, bottom_m_indices)
bottom_result_df = pd.DataFrame(list(bottom_result_dict.items()), columns=['Index', 'Path'])
bottom_result_df.to_csv(f'{SAVE_PATH}/bottom_results.csv', index=False)

all_result_dict = print_paths_for_indices(image_df, all_indices_descending)
all_result_df = pd.DataFrame(list(all_result_dict.items()), columns=['Index', 'Path'])
sorted_indices_all = sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)
sorted_prob_all = [prob[i] for i in sorted_indices_all]
all_result_df['prob'] = sorted_prob_all
all_result_df.to_csv(f'{SAVE_PATH}/all_results.csv', index=False)


# Visualize & Save Top/Bottom Ranked Images

def top_plot(df, name):
    """
    Displays the top 20 images per class from the dataframe (df).
    Saves the figure and CSV containing those images.

    Args:
        df (pd.DataFrame): Must have columns 'Index' and 'Path'.
        name (str): Used in filenames for saving (e.g., 'best_leverage_scores').
    """
    df['Path'] = df['Path'].astype(str)

    def extract_class(cleaned_path):
        """
        Attempt to parse out the class from the parent directory name.
        """
        try:
            filename = os.path.basename(os.path.dirname(cleaned_path))
            class_num = int(filename)
            return class_num
        except Exception as e:
            print(f"Error extracting class from path {cleaned_path}: {e}")
            return None

    df['Class'] = df['Path'].apply(extract_class)
    df = df.dropna(subset=['Class'])

    # Get the top 20 images per class
    top_images = df.groupby('Class').apply(lambda x: x.head(20)).reset_index(drop=True)

    # Save results to CSV
    if name == "best_leverage_scores":
        df.to_csv(f"{SAVE_PATH}/top_class_results.csv", index=False)
    else:
        df.to_csv(f"{SAVE_PATH}/bottom_class_results.csv", index=False)

    # Calculate how many classes exist
    num_classes = df['Class'].nunique()
    fig, axes = plt.subplots(num_classes, 20, figsize=(20, num_classes * 4))
    axes = axes.flatten()

    idx = 0
    for _, row in top_images.iterrows():
        if idx >= len(axes):
            break
        img_path = row['Path']
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Class {row['Class']}")
            axes[idx].axis('off')
            idx += 1
        except FileNotFoundError as e:
            print(f"File not found: {img_path}")

    plt.suptitle(name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{SAVE_PATH}/{name}_plot.png')
    plt.close()

# Plot & save top scoring images
top_plot(top_result_df, "best_leverage_scores")
# Plot & save bottom scoring images
top_plot(bottom_result_df, "worst_leverage_scores")
