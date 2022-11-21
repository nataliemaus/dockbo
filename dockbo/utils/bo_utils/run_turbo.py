import torch 
from dockbo.utils.bo_utils.bo_utils import (
    TurboState, 
    update_state, 
    generate_batch, 
    initialize_surrogate_model,
    update_surr_model,
)
from dockbo.utils.lightdock_torch_utils.lightdock_constants import (
    TH_DEVICE,
    TH_DTYPE,
)
from dockbo.utils.lightdock_torch_utils.feasibility_utils import is_valid_config


def run_turbo(
    bounding_box,
    check_validity_utils,
    max_n_bo_steps,
    bsz,
    learning_rte,
    verbose_config_opt,
    get_lightdock_score,
    max_n_tr_restarts,
    absolute_max_n_steps=10_000,
    min_lig_rest_value=0.05,
    min_rec_rest_value=0.05,
    init_n_epochs=60,
    update_n_epochs=10,
    n_init=100,
    init_w_known_pose=False,
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
        normalized = (x - lower_bounds1)/bound_range1 
        return normalized

    # Initialization data
    train_x = torch.rand(n_init, 27).cuda()  # random between 0 and 1 
    if init_w_known_pose:
        # add default/known pose (obtained w/ no rotaiton/transloation)# 1,0,0,0 --> identity rotation 
        default_config = torch.tensor([0.0]*3 + [1] + [0.0]*(27-4)) 
        default_config_normed = normalize(default_config.cuda())
        train_x = torch.cat((train_x, default_config_normed.unsqueeze(0)))

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

    # perc_restraints_only = False
    # if n_restraints > 0:
    #     perc_restraints_only = True 
    # reset_perc_restraints_only_at = n_restraints - 0.1 # // 2 

    train_y = []
    for x in train_x:
        e = get_lightdock_score(unnormalize(x) )['energy']
        train_y.append(e)
        # train_c_rec.append(perc_rec)
        # train_c_lig.append(perc_lig)

    train_y = torch.tensor(train_y).float().unsqueeze(-1)
    # train_c_rec = torch.tensor(train_c_rec).float().unsqueeze(-1)
    # train_c_lig = torch.tensor(train_c_lig).float().unsqueeze(-1)

    # Run TuRBO 
    num_restarts = 0 
    turbo_state = TurboState() 

    surr_model, mll = initialize_surrogate_model(train_x )
    # get surr model updated on data 
    surr_model = update_surr_model(
        surr_model,
        mll,
        train_x,
        train_y,
        init_n_epochs ,
        learning_rte
    )
    n_steps = 0
    n_productive_steps = 0
    # neg_energy = False 
    while n_productive_steps < max_n_bo_steps:   # and (num_restarts < max_n_tr_restarts)) or neg_energy:
        n_steps += 1
        if train_y.max().item() > 0.0:
            n_productive_steps += 1
        # get surr model updated on data  
        surr_model = update_surr_model(
            surr_model,
            mll,
            train_x,
            train_y,
            update_n_epochs,
            learning_rte
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
            lig_rest_model=None,
            rec_rest_model=None,
            min_lig_rest_value=min_lig_rest_value,
            min_rec_rest_value=min_rec_rest_value,
        )

        # make sure given x is valid (x,y,z loc falls outside of receptor)
        for x in x_next:
            assert is_valid_config(unnormalize(x), check_validity_utils)
        # compute scores for batch of candidates 
        y_next = []
        for x in x_next:
            e = get_lightdock_score(unnormalize(x))['energy']
            y_next.append(e)

        y_next = torch.tensor(y_next).float().unsqueeze(-1)
        # update data 
        train_x = torch.cat((train_x, x_next))
        train_y = torch.cat((train_y, y_next)) 

        # update turbo state 
        turbo_state = update_state(turbo_state, y_next)
        if turbo_state.restart_triggered: # if restart triggered, new state
            turbo_state = TurboState() 
            num_restarts += 1
        
        # valid_train_y = train_y[(train_c_rec > min_rec_rest_value) & (train_c_lig > min_lig_rest_value)] 
        if verbose_config_opt:
            print(f"N configs evaluated:{len(train_y)}")
            print(f'Best config score:{train_y.max().item()}')
        
        # neg_energy = False
        # if len(valid_train_y) == 0:
        #     neg_energy = True # keep looping, none valid 
        # elif valid_train_y.max().item() <= 0.0:
        #     neg_energy = True 
        # if train_y.max().item() <= 0.0:
        #     neg_energy = True 
        
        if n_steps > absolute_max_n_steps:
            print(f"absolute_max_n_steps = {absolute_max_n_steps} exceeded")
            break

    # valid_train_y = train_y[(train_c_rec > min_rec_rest_value) & (train_c_lig > min_lig_rest_value)] 
    # best_score = valid_train_y.max().item() 
    best_score = train_y.max().item() 
    # try:
    #     valid_train_x = train_x[(train_c_rec.squeeze() > min_rec_rest_value) & (train_c_lig.squeeze() > min_lig_rest_value)] 
    # except:
    #     import pdb 
    #     pdb.set_trace() 
    # best_config = unnormalize(valid_train_x[valid_train_y.argmax()].squeeze()).unsqueeze(0).detach().cpu()
    best_config = unnormalize(train_x[train_y.argmax()].squeeze()).unsqueeze(0).detach().cpu()

    return best_score, best_config 
