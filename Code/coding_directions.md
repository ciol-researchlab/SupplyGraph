# Using **SupplyGraph** Dataset for Temporal Graphs

This guide provides step-by-step directions for using the **SupplyGraph** dataset in building a homogeneous temporal graph using the *PyTorch Geometric Temporal* library. In this example, we will use production data as node features and connect nodes based on common production plants. This is just an example, we do not want to limit the applications by providing solution of a fixed problem.

### 1. Build the Graph
- **Nodes**: In this example, nodes represent products.
- **Edges**: Products produced in the same plant are connected by edges. The edges represent the commonality of production locations.
  
### 2. Install and Load the Temporal Graph Library
We will use the [*PyTorch Geometric Temporal*](https://pytorch-geometric-temporal.readthedocs.io/) library. You can install it using the following command:

```bash
!pip install torch-geometric-temporal
```

### 3. Transform Node Features
The raw dataset provides production data in CSV format, like this:

```
| Date               | SOS008L02P | SOS005L04P | SOS003L04P | ...
|--------------------|------------|------------|------------|----
| 2023-01-01 00:00:00| 14.83776   | 55.3472    | 0.0        | ...
| 2023-01-02 00:00:00| 14.92608   | 110.4      | 0.0        | ...
| 2023-01-03 00:00:00| 0.0        | 73.8208    | 0.0        | ...
```

We need to transform this into a format where products (nodes) are identified by numbers (0, 1, 2, etc.) and the date is removed:

```
| 0        | 1       | 2       | ...
|----------|---------|---------|----
| 14.83776 | 55.3472 | 0.0     | ...
| 14.92608 | 110.4   | 0.0     | ...
| 0.0      | 73.8208 | 0.0     | ...
```

- Replace product IDs with numerical values (e.g., 0, 1, 2).
- Remove the date column.
- Use only the numerical values of production data for node features.

You can also add additional temporal data, like delivery to distributors or factory issues, making the node features 3D or 4D.

### 4. Transform Output Data
The same transformation should be applied to output data (target data). Both input and output data are needed for temporal prediction.

### 5. Load the Data
Once the data has been transformed, load it for graph processing.

### 6. Load `StaticGraphTemporalSignal` Dataset Class
The `StaticGraphTemporalSignal` class from *PyTorch Geometric Temporal* is used to manage the temporal dataset. You will need to load your data into this class.

### 7. Build the Dataset
Now, build the dataset using the following:

```python
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

dataset = StaticGraphTemporalSignal(
    edge_index=edges_index,     # Encoded edge indices
    edge_weight=edges_f,        # Edge weights, or use np.ones((num_nodes, 1)) if none
    features=node_f,            # Node features, reshaped as needed
    targets=node_p              # Target values for prediction
)
```

- `edge_index`: A numpy array encoding the graph structure with edge connections.
- `edge_weight`: You can use attributes like production similarity as edge weights, or simply set them to 1 using `edges_f = np.ones((num_nodes, 1))`.
- `features`: The reshaped dataframe of node features. It should be reshaped as `node_f.T.reshape(temporal_length, num_nodes, dimension)`. In our case, the dimension is 1 because we are using only production data.
- `targets`: Set the corresponding output values here.

### 8. Splitting the Data
You can split the dataset into training and testing sets using:

```python
from torch_geometric_temporal.signal import temporal_signal_split

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
```

This will split 80% of the data for training and 20% for testing.

### 9. Define and Load a Model
We will use the `GCLSTM` model from the *PyTorch Geometric Temporal* library. Here is an example model class:

```python
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(node_features, 32, 1)   # Define the recurrent layer
        self.linear = torch.nn.Linear(32, 1)           # Final linear layer

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)  # Forward pass through GCLSTM
        h = F.relu(h)                                  # Apply ReLU activation
        h = self.linear(h)                             # Pass through linear layer
        return h
```

You can modify this model or use any other model available in the library. This model takes node features, edge indices, and edge weights as input.

### 10. Initialize the Model and Optimizer
Initialize the model with `node_features = 1` (since we are using only production data) and set the optimizer:

```python
model = RecurrentGCN(node_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

### 11. Train the Model
Now you can train the model using the following loop:

```python
from tqdm import tqdm

n_epoch = 100  # Number of epochs
for epoch in tqdm(range(n_epoch)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)  # Model prediction
        cost += torch.mean((y_hat - snapshot.y) ** 2)                       # MSE loss
    cost = cost / (time + 1)                                                # Average cost
    cost.backward()                                                         # Backpropagation
    optimizer.step()                                                        # Update weights
    optimizer.zero_grad()                                                   # Reset gradients
```

In this loop:
- The model is trained over several epochs.
- The Mean Squared Error (MSE) between the predicted output and actual target is calculated and minimized using gradient descent.

You can adjust the number of epochs and learning rate to fine-tune the training process.

### 12. Heterogeneous modeling
For heterogeneous modeling, [check this](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html)! Follow instructions, change code as necessary to build dataset and model.