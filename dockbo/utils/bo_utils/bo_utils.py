import sys
sys.path.append("../")
import math
import pdb
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.generation.sampling import MaxPosteriorSampling
import gpytorch 
from gpytorch.mlls import PredictiveLogLikelihood
from .ppgpr import GPModel 
from torch.utils.data import TensorDataset, DataLoader


@dataclass
class TurboState:
    dim: int = 27
    batch_size: int = 1
    length: float = 0.8 # 0.8 ... 
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: float = float("nan")  # Note: Post-initialized # 32... 
    success_counter: int = 0
    success_tolerance: int = 3  # 10 # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim ) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size=1,
    dtype=torch.float32,
    device=torch.device('cuda'),
):
    assert torch.all(torch.isfinite(Y))
    n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
    x_center = X[Y.squeeze().argmax(), :].clone()  
    weights = torch.ones_like(x_center)
    # Asume X is normalized to [0,1]
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # thompson samplling: 
    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
    tr_lb = tr_lb.cuda()
    tr_ub = tr_ub.cuda()
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
    mask = mask.cuda()

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand = X_cand.cuda()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points 
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False) 
    X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size)

    return X_next


def initialize_surrogate_model(train_x ):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
    n_init_points = min(1024, len(train_x))
    init_train_x = train_x[0:n_init_points] 
    model = GPModel(init_train_x.cuda(), likelihood=likelihood ).cuda()
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=init_train_x.size(-2))
    model = model.cuda()

    return model, mll 


def get_surr_model(
    train_x,
    train_y,
    n_epochs,
    learning_rte
):
    model, mll = initialize_surrogate_model(train_x )
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rte} ], lr=learning_rte)
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_x.cuda(), train_y.cuda()) 
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.squeeze().cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model

