import torch 
from dockbo.utils.bo_utils.bo_utils import TurboState, update_state, generate_batch, get_surr_model
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    TH_DEVICE,
    TH_DTYPE,
)
from dockbo.utils.lightdock_torch_utils.feasibility_utils import is_valid_config


def run_turbo(
    bounding_box,
    n_init,
    check_validity_utils,
    max_n_bo_steps,
    bsz,
    n_epochs,
    learning_rte,
    verbose_config_opt,
    get_lightdock_score,
    max_n_tr_restarts,
):
    # GOAL: Find optimal ld_sol (27 numbers) within bounds given by bounding_box
    #   which are the optimal configuraiton of ligand and protein to get max score 
    lower_bounds = bounding_box[:,0].cuda()
    upper_bounds = bounding_box[:,1].cuda()
    bound_range = (upper_bounds - lower_bounds).cuda() 

    # xs normalized 0 to 1, unnormalize to get lightdock scores 
    def unnormalize(x):
        if len(x.shape) > 1: # if batch dim included
            bound_range1 = bound_range.unsqueeze(0)
            lower_bounds1 = lower_bounds.unsqueeze(0)
        else:
            bound_range1 = bound_range
            lower_bounds1 = lower_bounds
        unnormalized = x.cuda()*bound_range1 + lower_bounds1

        return unnormalized 

    def normalize(x):
        if len(x.shape) > 1: # if batch dim included
            bound_range1 = bound_range.unsqueeze(0)
            lower_bounds1 = lower_bounds.unsqueeze(0)
        else:
            bound_range1 = bound_range
            lower_bounds1 = lower_bounds
        normed = (x.cuda() - lower_bounds1)/bound_range1
        return normed 

    # Initialization data
    train_x = torch.rand(n_init, 27)  # random between 0 and 1 

    # additionally initialize w/ best config from previous opt
    # if self.previous_best_config is not None: 
    #     train_x = torch.cat((train_x, normalize(self.previous_best_config))) 

    # remove invalid initial xs
    un_train_X = unnormalize(train_x).detach().cpu()
    valid_samples = []
    for ux in un_train_X:
        valid_samples.append(is_valid_config(ux, check_validity_utils))
    valid_samples = torch.tensor(valid_samples) # bool tensor 
    train_x = train_x[valid_samples]
    # catch case when all invalid, sample unntil at least one valid x is found  
    while len(train_x) == 0:
        new_x = torch.rand(1, 27) 
        if is_valid_config(unnormalize(new_x.squeeze()), check_validity_utils):
            train_x = new_x 

    train_y = [get_lightdock_score(unnormalize(x)) for x in train_x]
    train_y = torch.tensor(train_y).float().unsqueeze(-1)
    # print(f"max of init train y: {train_y.max()}")

    # Run TuRBO 
    num_restarts = 0 
    turbo_state = TurboState() 
    for _ in range(max_n_bo_steps): 
        # get surr model updated on data 
        surr_model = get_surr_model( 
            train_x=train_x,
            train_y=train_y,
            n_epochs=n_epochs,
            learning_rte=learning_rte,
        )
        # generate batch of candidates in trust region w/ thompson sampling
        x_next = generate_batch(
            state=turbo_state,
            model=surr_model,  # GP model
            X=train_x,  # Evaluated points on the domain [0, 1]^d
            Y=train_y,  # Function values
            batch_size=bsz,
            dtype=TH_DTYPE,
            device=TH_DEVICE,
            check_validity_utils=check_validity_utils,
            unnormalize_func=unnormalize,
        )

        # make sure given x is valid (x,y,z loc falls outside of receptor)
        for x in x_next:
            assert is_valid_config(unnormalize(x), check_validity_utils)
        # compute scores for batch of candidates 
        y_next = [get_lightdock_score(unnormalize(x)) for x in x_next]
        y_next = torch.tensor(y_next).float().unsqueeze(-1)
        # update data 
        train_x = torch.cat((train_x, x_next.detach().cpu()))
        train_y = torch.cat((train_y, y_next)) 
        # update turbo state 
        turbo_state = update_state(turbo_state, y_next)
        if turbo_state.restart_triggered: # if restart triggered, new state
            turbo_state = TurboState() 
            num_restarts += 1
        if verbose_config_opt:
            print(f'N configs evaluated:{len(train_y)}, best config score:{train_y.max().item()}')
        if num_restarts >= max_n_tr_restarts:
            break 

    best_score = train_y.max().item() 
    best_config = unnormalize(train_x[train_y.argmax()].squeeze()).unsqueeze(0).detach().cpu()

    return best_score, best_config 