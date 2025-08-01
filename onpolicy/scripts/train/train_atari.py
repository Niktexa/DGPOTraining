# #!/usr/bin/env python
# import sys
# import os
# import wandb
# import socket
# import setproctitle
# import numpy as np
# from pathlib import Path
# import torch
# import gymnasium as gym
# import ale_py
# from onpolicy.config import get_config
# from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
# from AtariVMAPDWrapper import AtariVMAPDWrapper 

# """Train script for Atari."""

# def make_train_env(all_args):
#     def get_env_fn(rank):
#         def init_env():
#             if all_args.env_name == "Atari":
#                 env = gym.make(all_args.game_name)
#                 env = AtariVMAPDWrapper(env, all_args.max_z, rank%all_args.max_z)
#             else:
#                 print("Can not support the " + all_args.env_name + "environment.")
#                 raise NotImplementedError
#             env.seed(all_args.seed + rank * 1000)
#             return env

#         return init_env

#     if all_args.n_rollout_threads == 1:
#         return ShareDummyVecEnv([get_env_fn(0)])
#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


# def make_eval_env(all_args):
#     def get_env_fn(rank):
#         def init_env():
#             if all_args.env_name == "Atari":
#                 env = gym.make(all_args.game_name)
#                 env = AtariVMAPDWrapper(env, all_args.max_z, rank%all_args.max_z)
#             else:
#                 print("Can not support the " + all_args.env_name + "environment.")
#                 raise NotImplementedError
#             env.seed(all_args.seed * 50000 + rank * 10000)
#             return env

#         return init_env

#     if all_args.n_eval_rollout_threads == 1:
#         return ShareDummyVecEnv([get_env_fn(0)])
#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


# def parse_args(args, parser):
#     parser.add_argument('--game_name', type=str, default='PongNoFrameskip-v4', help="Which Atari game to run on")
#     parser.add_argument('--num_agents', type=int, default=1, help="number of players")

#     all_args = parser.parse_known_args(args)[0]
#     return all_args


# def main(args):
#     parser = get_config()
#     all_args = parse_args(args, parser)    

#     assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")

#     # cuda
#     if all_args.cuda and torch.cuda.is_available():
#         print("choose to use gpu...")
#         device = torch.device("cuda:0")
#         torch.set_num_threads(all_args.n_training_threads)
#         if all_args.cuda_deterministic:
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True
#     else:
#         print("choose to use cpu...")
#         device = torch.device("cpu")
#         torch.set_num_threads(all_args.n_training_threads)

#     run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
#                        0] + "/results") / all_args.env_name / all_args.game_name / all_args.algorithm_name / all_args.experiment_name
#     if not run_dir.exists():
#         os.makedirs(str(run_dir))

#     if all_args.use_wandb:
#         run = wandb.init(config=all_args,
#                          project="exp_result",
#                          entity=all_args.user_name,
#                          notes=socket.gethostname(),
#                          name=str(all_args.algorithm_name) + "_" +
#                               str(all_args.game_name) +
#                               "_seed" + str(all_args.seed),
#                          group=all_args.game_name,
#                          dir=str(run_dir),
#                          job_type="training",
#                          reinit=True)
#     else:
#         if not run_dir.exists():
#             curr_run = 'run1'
#         else:
#             exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
#                              str(folder.name).startswith('run')]
#             if len(exst_run_nums) == 0:
#                 curr_run = 'run1'
#             else:
#                 curr_run = 'run%i' % (max(exst_run_nums) + 1)
#         run_dir = run_dir / curr_run
#         if not run_dir.exists():
#             os.makedirs(str(run_dir))

#     setproctitle.setproctitle(
#         str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
#             all_args.user_name))

#     # seed
#     torch.manual_seed(all_args.seed)
#     torch.cuda.manual_seed_all(all_args.seed)
#     np.random.seed(all_args.seed)

#     print(f"🚀 CREATING {all_args.n_rollout_threads} PARALLEL ENVIRONMENTS...")
#     # env
#     envs = make_train_env(all_args)
#     eval_envs = make_eval_env(all_args) if all_args.use_eval else None
#     print("✅ ENVIRONMENTS CREATED SUCCESSFULLY!")
    
#     num_agents = all_args.num_agents
#     print(f"📊 Number of agents: {num_agents}")

#     config = {
#         "all_args": all_args,
#         "envs": envs,
#         "eval_envs": eval_envs,
#         "num_agents": num_agents,
#         "device": device,
#         "run_dir": run_dir
#     }

#     # run experiments
#     if all_args.share_policy:
#         from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
#     else:
#         raise NotImplementedError

#     print("🧠 INITIALIZING DGPO RUNNER...")
#     runner = Runner(config)
#     print("✅ DGPO RUNNER INITIALIZED!")
    
#     print("🏃 STARTING TRAINING LOOP...")
#     runner.run()

#     print(f"div_thresh: {all_args.div_thresh}")
#     print(f"rex_thresh: {all_args.rex_thresh}")  
#     print(f"alpha_rex: {all_args.alpha_rex}")
#     print(f"alpha_div: {all_args.alpha_div}")

#     # post process
#     envs.close()
#     if all_args.use_eval and eval_envs is not envs:
#         eval_envs.close()

#     if all_args.use_wandb:
#         run.finish()
#     else:
#         runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
#         runner.writter.close()


# if __name__ == "__main__":
#     import sys
#     # sys.argv = [
#     #     'train_atari.py', 
#     #     '--env_name', 'Atari', 
#     #     '--game_name', 'ALE/Pong-v5', 
#     #     '--max_z', '2', 
#     #     '--n_rollout_threads', '8',
#     #     '--episode_length', '25',        
#     #     '--ppo_epoch', '2',             
#     #     '--num_mini_batch', '2',        
#     #     '--vmapd_warmup', '2',
#     #     '--num_env_steps', '10000',     # 50 episodes
#     #     '--experiment_name', 'pong_v3', 
#     #     '--seed', '42'                  # Different seed for variety
#     # ]
#     sys.argv = [
#         'train_atari.py', 
#         '--env_name', 'Atari', 
#         '--game_name', 'ALE/Pong-v5', 
#         '--max_z', '2',                    # Table 4: nz = 2
#         '--div_thresh', '-0.105',          # Table 4: log(0.9)
#         '--rex_thresh', '0.8',             # Table 4: 0.8
#         '--alpha_rex', '1.0',              # Not specified, reasonable default
#         '--alpha_div', '1.0',              # Not specified, reasonable default
#         '--n_rollout_threads', '128',      # Table 1: Atari = 128
#         '--episode_length', '400',         # Table 1: Atari = 400
#         '--ppo_epoch', '128',              # Table 1: Atari = 128
#         '--num_mini_batch', '32',          # Reasonable for 128 parallel envs
#         '--num_env_steps', '50000000',     # 50M steps for full paper reproduction
#         '--experiment_name', 'pong_paper_exact',
#         '--seed', '42'
#     ]
#     main(sys.argv[1:])



#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import gymnasium as gym
import ale_py
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from AtariVMAPDWrapper import AtariVMAPDWrapper 
from tensorboard_logger import create_dgpo_logger

"""Train script for Atari with TensorBoard monitoring."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Atari":
                env = gym.make(all_args.game_name)
                env = AtariVMAPDWrapper(env, all_args.max_z, rank%all_args.max_z)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Atari":
                env = gym.make(all_args.game_name)
                env = AtariVMAPDWrapper(env, all_args.max_z, rank%all_args.max_z)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
 

def parse_args(args, parser):
    parser.add_argument('--game_name', type=str, default='PongNoFrameskip-v4', help="Which Atari game to run on")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")

    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)    

    assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.game_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Create TensorBoard Logger
    tb_logger = create_dgpo_logger(str(run_dir), all_args.experiment_name, all_args.max_z)

    
    # Log hyperparameters
    hparams = {
        'div_thresh': getattr(all_args, 'div_thresh', 'unknown'),
        'rex_thresh': getattr(all_args, 'rex_thresh', 'unknown'),
        'alpha_rex': getattr(all_args, 'alpha_rex', 'unknown'),
        'alpha_div': getattr(all_args, 'alpha_div', 'unknown'),
        'max_z': all_args.max_z,
        'n_rollout_threads': all_args.n_rollout_threads,
        'episode_length': all_args.episode_length,
        'ppo_epoch': all_args.ppo_epoch,
        'lr': getattr(all_args, 'lr', 'unknown'),
        'seed': all_args.seed
    }
    tb_logger.log_hyperparameters(hparams)
    
    print(f"TensorBoard logging enabled!")
    print(f"To view dashboard: tensorboard --logdir {run_dir}/tensorboard")
    print(f"Then open: http://localhost:6006")

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="exp_result",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.game_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.game_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    print(f"🚀 CREATING {all_args.n_rollout_threads} PARALLEL ENVIRONMENTS...")
    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    print("ENVIRONMENTS CREATED SUCCESSFULLY!")
    
    num_agents = all_args.num_agents
    print(f"Number of agents: {num_agents}")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "tb_logger": tb_logger  # ← NEW: Pass logger to runner
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
    else:
        raise NotImplementedError

    print("INITIALIZING DGPO RUNNER...")
    runner = Runner(config)
    print("DGPO RUNNER INITIALIZED")
    
    print("STARTING TRAINING LOOP...")
    print(f"Monitor progress: tensorboard --logdir {run_dir}/tensorboard")
    
    try:
        runner.run()
    finally:
        # Always close TensorBoard logger
        tb_logger.close()

    # Debug hyperparameter values
    print(f"div_thresh: {getattr(all_args, 'div_thresh', 'NOT FOUND')}")
    print(f"rex_thresh: {getattr(all_args, 'rex_thresh', 'NOT FOUND')}")  
    print(f"alpha_rex: {getattr(all_args, 'alpha_rex', 'NOT FOUND')}")
    print(f"alpha_div: {getattr(all_args, 'alpha_div', 'NOT FOUND')}")

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    import sys
    sys.argv = [
        'train_atari.py',
        '--env_name', 'Atari',
        '--game_name', 'ALE/Pong-v5',
        '--max_z', '2',                    # ✅ Table 4
        '--div_thresh', '0.9',             # ✅ Table 4  
        '--rex_thresh', '0.8',             # ✅ Table 4
        '--alpha_rex', '1.0',              
        '--alpha_div', '1.0',              
        '--n_rollout_threads', '16',       # ✅ Table 1: Atari = 16
        '--episode_length', '128',         # ✅ Table 1: Atari = 128  
        '--ppo_epoch', '4',                # ✅ Table 1: Atari = 4
        '--num_mini_batch', '4',           # Reasonable for 16 threads
        '--num_env_steps', '50000000',     
        '--experiment_name', 'pong_paper_exact_final',
        '--seed', '42'
    ]
    main(sys.argv[1:])





    # sys.argv = [
    #     'train_atari.py', 
    #     '--env_name', 'Atari', 
    #     '--game_name', 'ALE/Pong-v5', 
    #     '--max_z', '2',                    # Table 4: nz = 2
    #     '--div_thresh', '.9',          # Table 4: log(0.9)
    #     '--rex_thresh', '0.8',             # Table 4: 0.8
    #     '--alpha_rex', '1.0',              # Not specified, reasonable default
    #     '--alpha_div', '1.0',              # Not specified, reasonable default
    #     '--n_rollout_threads', '128',      # Table 1: Atari = 128
    #     '--episode_length', '400',         # Table 1: Atari = 400
    #     '--ppo_epoch', '128',              # Table 1: Atari = 128
    #     '--num_mini_batch', '32',          # Reasonable for 128 parallel envs
    #     '--num_env_steps', '50000000',     # 50M steps for full paper reproduction
    #     '--experiment_name', 'pong_paper_exact',
    #     '--seed', '42'
    # ]


    # sys.argv = [
    #         'train_atari.py', 
    #         '--env_name', 'Atari', 
    #         '--game_name', 'ALE/Pong-v5', 
    #         '--max_z', '2',
    #         '--div_thresh', '.9',
    #         '--rex_thresh', '0.8',
    #         '--alpha_rex', '1.0',
    #         '--alpha_div', '1.0',
    #         '--n_rollout_threads', '8',        # ← Reduced: 8 envs instead of 128
    #         '--episode_length', '25',          # ← Reduced: 25 steps instead of 400
    #         '--ppo_epoch', '2',                # ← Reduced: 2 epochs instead of 128
    #         '--num_mini_batch', '2',           # ← Reduced: 2 batches instead of 32
    #         '--num_env_steps', '10000',        # ← Reduced: ~50 episodes
    #         '--experiment_name', 'tensorboard_test-v2',
    #         '--seed', '42'
    #     ]

