# import numpy as np
# import copy
# from gym import spaces

# class AtariVMAPDWrapper(object):
#     # def __init__(self, env, max_z, fix_z):
#     #     super().__init__()
#     #     self.env = env
#     #     self.max_z = max_z
#     #     self.fix_z = fix_z
#     #     self.cur_z = -1
#     #     self.num_agents = 1  # Atari is single agent
        
#     #     # Create spaces for single agent (but keep list format for compatibility)
#     #     self.z_space = [spaces.Discrete(self.max_z)]
        
#     #     # IMPORTANT: Reset environment first to get actual observation shape
#     #     temp_obs = self.env.reset()
#     #     if isinstance(temp_obs, tuple):
#     #         actual_obs = temp_obs[0]
#     #     else:
#     #         actual_obs = temp_obs
        
#     #     # Calculate flattened size based on ACTUAL observation + z conditioning
#     #     z_vec = np.eye(self.max_z)[0]  # dummy z vector
#     #     obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(actual_obs.shape[1], axis=1).repeat(actual_obs.shape[2], axis=2), actual_obs], axis=0)
#     #     flattened_size = obs_with_z.flatten().shape[0]
        
#     #     self.observation_space = [spaces.Box(low=0, high=255, shape=(flattened_size,), dtype=np.uint8)]
#     #     self.share_observation_space = copy.deepcopy(self.observation_space)
#     #     self.action_space = [self.env.action_space]
        
#     #     # Z observation spaces (for DGPO)
#     #     self.z_obs_space = copy.deepcopy(self.observation_space)
#     #     self.z_local_obs_space = copy.deepcopy(self.observation_space)

#     def __init__(self, env, max_z, fix_z):
#         super().__init__()
#         self.env = env
#         self.max_z = max_z
#         self.fix_z = fix_z
#         self.cur_z = -1
#         self.num_agents = 1  # Atari is single agent
        
#         # Create spaces for single agent (but keep list format for compatibility)
#         self.z_space = [spaces.Discrete(self.max_z)]
        
#         # IMPORTANT: Reset environment first to get actual observation shape
#         temp_obs = self.env.reset()
#         if isinstance(temp_obs, tuple):
#             actual_obs = temp_obs[0]
#         else:
#             actual_obs = temp_obs
        
#         # DEBUG: Print actual shapes
#         print(f"DEBUG: Original obs shape: {actual_obs.shape}")
#         print(f"DEBUG: Original obs size: {np.prod(actual_obs.shape)}")
        
#         # Calculate flattened size based on ACTUAL observation + z conditioning
#         z_vec = np.eye(self.max_z)[0]  # dummy z vector
#         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(actual_obs.shape[1], axis=1).repeat(actual_obs.shape[2], axis=2), actual_obs], axis=0)
#         flattened_size = obs_with_z.flatten().shape[0]
        
#         print(f"DEBUG: Final flattened size: {flattened_size}")

#         # FIX: Discriminator actually receives 101754, so initialize for that
#         actual_discriminator_size = 101754  # What discriminator actually gets
        
#         self.observation_space = [spaces.Box(low=0, high=255, shape=(actual_discriminator_size,), dtype=np.uint8)]
        
#         self.share_observation_space = copy.deepcopy(self.observation_space)
#         self.action_space = [self.env.action_space]
        
#         # Z observation spaces (for DGPO)
#         self.z_obs_space = copy.deepcopy(self.observation_space)
#         self.z_local_obs_space = copy.deepcopy(self.observation_space)


#     def reset(self, fix_z=None):
#         if fix_z is not None:
#             self.cur_z = fix_z
#         elif self.fix_z is not None:
#             self.cur_z = self.fix_z
#         else:
#             self.cur_z = np.random.randint(self.max_z)
        
#         # Handle both old and new gymnasium reset() formats
#         result = self.env.reset()
#         if isinstance(result, tuple):
#             obs, info = result  # New gymnasium format
#         else:
#             obs = result  # Old gym format
        
#         z_vec = np.eye(self.max_z)[self.cur_z]
        
#         # Add z to the first channel of the observation
#         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
#         # Create proper available_actions for Atari (all actions always available)
#         num_actions = self.env.action_space.n
#         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
#         obs_with_z_flat = obs_with_z.flatten()
            
#         target_size = 101754
#         obs_with_z_flat = obs_with_z_flat[:target_size]
        
#         print(f"RESET DEBUG: Returning obs size: {obs_with_z_flat.shape}")
        
#         return [obs_with_z_flat], [obs_with_z_flat], available_actions


#     def step(self, actions):
#         # Convert action to integer (ALE expects int, not array)
#         if isinstance(actions, list):
#             action = int(actions[0])
#         else:
#             action = int(actions)
        
#         # Handle both old and new gymnasium step() formats
#         result = self.env.step(action)
#         if len(result) == 4:
#             obs, reward, done, info = result  # Old format
#         else:
#             obs, reward, terminated, truncated, info = result  # New format
#             done = terminated or truncated
        
#         # Structure info for multi-agent compatibility
#         agent_info = dict(info)
#         agent_info['bad_transition'] = False
#         multi_agent_info = [agent_info]
        
#         z_vec = np.eye(self.max_z)[self.cur_z]
#         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
#         # Create proper available_actions for Atari
#         num_actions = self.env.action_space.n
#         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
#         obs_with_z_flat = obs_with_z.flatten()

#         target_size = 101754
#         obs_with_z_flat = obs_with_z_flat[:target_size]
        
#         print(f"WRAPPER DEBUG: Returning obs size: {obs_with_z_flat.shape}")
        

        
#         reward_shaped = np.array([reward], dtype=np.float32)
#         done_shaped = [done]
        
#         return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions

#     def seed(self, seed):
#         if hasattr(self.env, 'seed'):
#             return self.env.seed(seed)

#     def close(self):
#         self.env.close()


import numpy as np
import copy
from gym import spaces

class AtariVMAPDWrapper(object):
    def __init__(self, env, max_z, fix_z):
        super().__init__()
        self.env = env
        self.max_z = max_z
        self.fix_z = fix_z
        self.cur_z = -1
        self.num_agents = 1  # Atari is single agent
        
        # Create spaces for single agent (but keep list format for compatibility)
        self.z_space = [spaces.Discrete(self.max_z)]
        
        # IMPORTANT: Reset environment first to get actual observation shape
        temp_obs = self.env.reset()
        if isinstance(temp_obs, tuple):
            actual_obs = temp_obs[0]
        else:
            actual_obs = temp_obs
        
        # DEBUG: Print actual shapes
        # print(f"DEBUG: Original obs shape: {actual_obs.shape}")
        # print(f"DEBUG: Original obs size: {np.prod(actual_obs.shape)}")
        
        # Calculate flattened size based on ACTUAL observation + z conditioning
        z_vec = np.eye(self.max_z)[0]  # dummy z vector
        obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(actual_obs.shape[1], axis=1).repeat(actual_obs.shape[2], axis=2), actual_obs], axis=0)
        flattened_size = obs_with_z.flatten().shape[0]
        
        # print(f"DEBUG: Natural flattened size: {flattened_size}")

        # Now that discriminator handles z-stripping internally, use natural size
        self.target_obs_size = flattened_size  # Your wrapper provides full size (101760)
        
        # print(f"DEBUG: Wrapper will provide: {flattened_size}")
        # print(f"DEBUG: Discriminator will strip {self.max_z} elements internally")
        
        # CRITICAL FIX: Use the natural size since discriminator now handles z-stripping
        self.observation_space = [spaces.Box(low=0, high=255, shape=(flattened_size,), dtype=np.uint8)]
        self.share_observation_space = copy.deepcopy(self.observation_space)
        self.action_space = [self.env.action_space]
        
        # Z observation spaces (for DGPO)
        self.z_obs_space = copy.deepcopy(self.observation_space)
        self.z_local_obs_space = copy.deepcopy(self.observation_space)

    def reset(self, fix_z=None):
        if fix_z is not None:
            self.cur_z = fix_z
        elif self.fix_z is not None:
            self.cur_z = self.fix_z
        else:
            self.cur_z = np.random.randint(self.max_z)
        
        # Handle both old and new gymnasium reset() formats
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result  # New gymnasium format
        else:
            obs = result  # Old gym format
        
        z_vec = np.eye(self.max_z)[self.cur_z]
        
        # Add z to the first channel of the observation
        obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
        # Create proper available_actions for Atari (all actions always available)
        num_actions = self.env.action_space.n
        available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
        obs_with_z_flat = obs_with_z.flatten()
        
        # Provide the natural full observation with z-vector (101760)
        # The discriminator will handle z-stripping internally
        if len(obs_with_z_flat) > self.target_obs_size:
            obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
        elif len(obs_with_z_flat) < self.target_obs_size:
            # Pad with zeros if somehow too small
            padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
            obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
        # print(f"RESET DEBUG: Returning obs size: {obs_with_z_flat.shape}")
        
        return [obs_with_z_flat], [obs_with_z_flat], available_actions

    def step(self, actions):
        # Convert action to integer (ALE expects int, not array)
        if isinstance(actions, list):
            action = int(actions[0])
        else:
            action = int(actions)
        
        # Handle both old and new gymnasium step() formats
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result  # Old format
        else:
            obs, reward, terminated, truncated, info = result  # New format
            done = terminated or truncated
        
        # Structure info for multi-agent compatibility
        agent_info = dict(info)
        agent_info['bad_transition'] = False
        multi_agent_info = [agent_info]
        
        z_vec = np.eye(self.max_z)[self.cur_z]
        obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
        # Create proper available_actions for Atari
        num_actions = self.env.action_space.n
        available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
        obs_with_z_flat = obs_with_z.flatten()

        # Provide the natural full observation with z-vector (101760)
        # The discriminator will handle z-stripping internally
        if len(obs_with_z_flat) > self.target_obs_size:
            obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
        elif len(obs_with_z_flat) < self.target_obs_size:
            # Pad with zeros if somehow too small
            padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
            obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
        # print(f"WRAPPER DEBUG: Returning obs size: {obs_with_z_flat.shape}")
        
        reward_shaped = np.array([reward], dtype=np.float32)
        done_shaped = [done]
        
        return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def close(self):
        self.env.close()