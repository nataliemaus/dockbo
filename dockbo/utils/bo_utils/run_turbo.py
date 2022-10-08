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
        if verbose_config_opt:
            print(f'N configs evaluated:{len(train_y)}, best config score:{train_y.max().item()}')
        
        # if train_y.max().item() > 100:# 2_000_000:
        #     import pdb 
        #     pdb.set_trace() 
            # 0/6625=0.0000 of ligand atoms within convex hull of receptor.
            # 0/6625=0.0000 of ligand atoms within 3.0 of any receptor atom.
            # Final score: 2095398.25
            # best_config = unnormalize(train_x[train_y.argmax()].squeeze()).unsqueeze(0).detach().cpu()
            #         tensor([[ 6.6545, 21.6246, 21.1632, -0.0414,  0.0878,  0.1561,  0.0299,  0.6081,
            # 0.6224,  0.7779,  0.7415,  0.7350,  0.7381,  0.3700,  0.4001,  0.6139,
            # 0.0349,  0.6773,  0.5559,  0.2530,  0.5940,  0.0309,  0.2715,  0.7529,
            # 0.5724,  0.6441,  0.2677]])
            # check_score = self.get_lightdock_score(best_config)
        
        # max of init train y: -427.98565673828125
    best_score = train_y.max().item() 
    best_config = unnormalize(train_x[train_y.argmax()].squeeze()).unsqueeze(0).detach().cpu()

    return best_score, best_config 