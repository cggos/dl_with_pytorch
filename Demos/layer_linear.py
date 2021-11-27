import torch
import torch.nn as nn

if __name__ == "__main__":
    in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

    weight_matrix = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=torch.float32)

    out_features = weight_matrix.matmul(in_features)

    print(f'out_features: {out_features}')

    fc = nn.Linear(4, 3, False)
    fc.weight = nn.Parameter(weight_matrix)
    out_features = fc(in_features)

    print(f'out_features: {out_features}')
