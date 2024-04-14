import torch

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

# Copied this formula from the existing codebase
def equal_earth_projection(Location):
    lat_rad = torch.deg2rad(Location[:, 0])
    long_rad = torch.deg2rad(Location[:, 1])

    theta = torch.asin((torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(lat_rad))
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    
    x = (2 * torch.sqrt(torch.tensor(3.0)) * long_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    
    return (torch.stack((x, y), dim=1) * SF) / 180