# Non-Linear Leverage Scores Regression
import torch, datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os, random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1) Reproducibility Utilities

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# 3) Define a Single-Neuron Model with Custom Swish Activation

def output_activation(t, nn=2, c1=1.0, c2=2.0, zeta=1.0):
    # return 1 / (1 + np.exp(-x))
    return t * (np.sqrt(c1) + (np.sqrt(c2) - np.sqrt(c1)) / (1 + np.exp(-zeta * t)))
    
def swish_torch(x, c1=1.0, c2=2.0, zeta=1.0):
    """
    Custom Swish with the formula:
      phi(t) = t * [ sqrt(c1) + ( sqrt(c2) - sqrt(c1) ) / (1 + e^(-zeta * t)) ]
    """
    return x * (
        torch.sqrt(torch.tensor(c1)) 
        + (torch.sqrt(torch.tensor(c2)) - torch.sqrt(torch.tensor(c1))) 
        / (1.0 + torch.exp(-zeta * x))
    )


# Single-Neuron Model

class SingleNeuronSwish(nn.Module):
    def __init__(self, input_size):
        super(SingleNeuronSwish, self).__init__()
        # Single neuron: 1 linear layer with (input_size -> 1) dimension
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # z = w^T x + b
        z = self.linear(x)
        # Swish activation
        return swish_torch(z)

# Set the seed for reproducibility
set_seed(42)
f_star_test = False
f_star_num_cal = True
input_size = None # Set Size
time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(time, flush=True)
SAVE_PATH = ''
MODEL_PATH = ''
DATA_PATH = ''
os.mkdir(SAVE_PATH)

# Load Model & Weights

# Load the model
model = SingleNeuronSwish(input_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Extract weights and bias
weights = model.linear.weight.data
bias = model.linear.bias.data

# Convert to NumPy array
weights_np = weights.detach().numpy()
bias_np = bias.detach().numpy()

# Print the weights and bias
# print("Weights (NumPy):", weights_np)
# print("Bias (NumPy):", bias_np)

theta = weights_np[0]
theta  = np.append(theta, bias_np[0]) # Adding Weights and Bias

theta_f_star_numerical = theta
theta_f_star_numerical = np.array(theta_f_star_numerical)

print(theta_f_star_numerical, theta_f_star_numerical.shape, flush=True)

def f1(theta, x, y):
    """
    Compute f(θ, x_i) = swish((b x)) - y_i
    """
    z = theta[:-1] @ x + theta[-1]
    y1 = swish_torch(z)  - y

    return y1.squeeze()

def compute_f_star(theta, x, y, N=5000):
    """
    Compute F*(θ, x) = ∫₀¹ ∂/∂θ f(tθ, x) dt numerically

    Parameters:
    - theta: Model Param
    - x: Input vector
    - N: Number of intervals for numerical integration

    Returns:
    - f_star: the computed F*(θ, x)
    """

    # Ensure theta requires gradients
    theta = theta.clone().detach().float().requires_grad_(True)
    x = x.clone().detach().float()
    f_star = torch.zeros_like(theta)

    # Discretize t from 0 to 1
    t_values = torch.linspace(0, 1, N + 1, dtype=torch.float32)[1:]  # Exclude t=0 to avoid zero gradients

    for t in t_values:
        t = t.item()  # Get scalar value from tensor
        # Scale theta by t
        theta_t = t * theta

        # Compute f(θ_t, x)
        y_t = f1(theta_t, x, y)

        # Compute gradient of y_t with respect to theta
        grad_theta = torch.autograd.grad(outputs=y_t, inputs=theta, retain_graph=False, create_graph=False)[0]

        # Accumulate the gradients
        f_star += grad_theta / N

    return f_star


if f_star_test: # Testing (ignore)

    # Your input vector x
    x = torch.randn(input_size)  # Replace with your actual input

    # Compute F*(theta, x)
    f_star_numerical = compute_f_star(theta_f_star_numerical, x, N=input_size)

    print("Computed F*(theta, x):", f_star_numerical, flush=True)


def phi(x):
    return output_activation(x)

def f_star_row(theta, x= torch.rand(100).numpy(), y = 0):
    # Compute ⟨θ, x⟩ (dot product)
    dot_product = np.dot(theta[:-1], x) + theta[-1]  # Include the bias term (theta[-1])
        
    if abs(dot_product) == 0: return np.zeros(len(theta)) ## Handling bias term via partial differentiation (hence 1 at last)
        
    # Compute F*(θ, x)
    phi_diff = phi(dot_product) - y
    scale = phi_diff / dot_product
    f_star_x = scale * x
    f_star_x = torch.cat((f_star_x, torch.tensor([1], dtype=f_star_x.dtype, device=f_star_x.device)), dim=0)

    return f_star_x


df = pd.read_csv(DATA_PATH)
X, y= df.drop(columns=["y"]), df["y"]

# f_star_matrix = np.array(f_star_matrix)
f_star_matrix_ls = []
path_dict = {}

for x in range(len(X)):
    
    if x%100==0: print(x, flush=True)

    if f_star_num_cal:
        f_star_numerical = compute_f_star(torch.from_numpy(theta_f_star_numerical).float(), torch.from_numpy(X.iloc[x].to_numpy()).float(), y[x], N=5000)  # Calculated via integration (definition)
        f_star_numerical = f_star_numerical.numpy()
        f_star_numerical = np.append(f_star_numerical, phi(0)) # adding (\phi(0))
        f_star_matrix_ls.append(f_star_numerical) 
    else:
        f_star_numerical = f_star_row(torch.from_numpy(theta_f_star_numerical).float(), torch.from_numpy(X.iloc[x].to_numpy()).float(), y[x])    
        f_star_numerical = f_star_numerical.numpy()
        f_star_numerical = np.append(f_star_numerical, phi(0)) # adding (\phi(0))
        f_star_matrix_ls.append(f_star_numerical)

    row_path = X.iloc[x].tolist()
    row_path.append(y[x])
    path_dict[len(path_dict)] = row_path

# print(path_dict, flush=True)
col_list = df.columns.tolist()
path_df = pd.DataFrame(list(path_dict.values()), columns=col_list)  
path_df.to_csv(SAVE_PATH)  
f_star_matrix_ls = np.array(f_star_matrix_ls)
print(f_star_matrix_ls.shape, flush=True)
df_f_star = pd.DataFrame(f_star_matrix_ls)
df_f_star.to_csv(SAVE_PATH)

for samp_method in ['sls', 'rns']: # calculate scores (F*) for each method
    nlls = np.zeros(len(f_star_matrix_ls)) # calculate statistical leverage scores

    if samp_method == 'sls':
        f_star_matrix_ls_XTX = np.dot(np.transpose(f_star_matrix_ls), f_star_matrix_ls)

        rank = np.linalg.matrix_rank(f_star_matrix_ls_XTX)
        print("Rank of f_star_matrix_ls_XTX:", rank, flush=True)
        print("Shape of f_star_matrix_ls_XTX:", f_star_matrix_ls_XTX.shape, flush=True)

        zero_rows = np.where(~f_star_matrix_ls.any(axis=1))[0]
        zero_cols = np.where(~f_star_matrix_ls.any(axis=0))[0]
        print("Zero rows:", zero_rows, flush=True)
        print("Zero columns:", zero_cols, flush=True)

        condition_number = np.linalg.cond(f_star_matrix_ls_XTX)
        print("Condition number of f_star_matrix_ls_XTX:", condition_number, flush=True)


        f_star_matrix_ls_XTX_inv = np.linalg.pinv(f_star_matrix_ls_XTX)
        print(len(nlls), flush=True)
        for index in range(len(nlls)):
            x = f_star_matrix_ls[index]
            nlls[index] = np.dot(np.dot(np.transpose(x), f_star_matrix_ls_XTX_inv), x)  
    else:  # Calculate row-norm scores normalized by Frobenius norm
        frobenius_norm_squared = np.linalg.norm(f_star_matrix_ls, 'fro') ** 2  # Frobenius norm squared
        for index in range(len(nlls)):
            row = f_star_matrix_ls[index]
            nlls[index] = (np.linalg.norm(row) ** 2) / frobenius_norm_squared  # Normalize by Frobenius norm

    prob = nlls / np.sum(nlls)
    print('Probability:',prob, flush=True)
    print(np.sum(prob), flush=True)

    import matplotlib.pyplot as plt
    plt.plot(prob)
    plt.savefig(SAVE_PATH)
    plt.close()

    # Define the file name
    output_file = f"{SAVE_PATH}/{samp_method}_scores.txt"
    with open(output_file, "w") as file:
        for score in prob:
            file.write(f"{score}\n")  # Write each score on a new line
    print(f"List saved to {output_file}", flush=True)

    def top_m_indices_and_values(lst, m):
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
        top_m_indices = sorted_indices[:m]
        top_m_values = [lst[i] for i in top_m_indices]
        return top_m_indices, top_m_values
    
    all_indices_descending, all_values_descending = top_m_indices_and_values(prob, len(prob))
    adjusted_indices = [i - 1 for i in all_indices_descending]
    reordered_df = path_df.iloc[adjusted_indices].reset_index(drop=True)
    reordered_df['values'] = all_values_descending
    reordered_df.to_csv(f"{SAVE_PATH}/ordered_df_{samp_method}.csv", index=False)