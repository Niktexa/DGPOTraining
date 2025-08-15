# import numpy as np
# import copy
# from gym import spaces

# class AtariVMAPDWrapper(object):
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
#         # print(f"DEBUG: Original obs shape: {actual_obs.shape}")
#         # print(f"DEBUG: Original obs size: {np.prod(actual_obs.shape)}")
        
#         # Calculate flattened size based on ACTUAL observation + z conditioning
#         z_vec = np.eye(self.max_z)[0]  # dummy z vector
#         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(actual_obs.shape[1], axis=1).repeat(actual_obs.shape[2], axis=2), actual_obs], axis=0)
#         flattened_size = obs_with_z.flatten().shape[0]
        
#         # print(f"DEBUG: Natural flattened size: {flattened_size}")

#         # Now that discriminator handles z-stripping internally, use natural size
#         self.target_obs_size = flattened_size  # Your wrapper provides full size (101760)
        
#         # print(f"DEBUG: Wrapper will provide: {flattened_size}")
#         # print(f"DEBUG: Discriminator will strip {self.max_z} elements internally")
        
#         # CRITICAL FIX: Use the natural size since discriminator now handles z-stripping
#         self.observation_space = [spaces.Box(low=0, high=255, shape=(flattened_size,), dtype=np.uint8)]
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
        
#         # Provide the natural full observation with z-vector (101760)
#         # The discriminator will handle z-stripping internally
#         if len(obs_with_z_flat) > self.target_obs_size:
#             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
#         elif len(obs_with_z_flat) < self.target_obs_size:
#             # Pad with zeros if somehow too small
#             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
#             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
#         # print(f"RESET DEBUG: Returning obs size: {obs_with_z_flat.shape}")
        
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

#         # Provide the natural full observation with z-vector (101760)
#         # The discriminator will handle z-stripping internally
#         if len(obs_with_z_flat) > self.target_obs_size:
#             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
#         elif len(obs_with_z_flat) < self.target_obs_size:
#             # Pad with zeros if somehow too small
#             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
#             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
#         # print(f"WRAPPER DEBUG: Returning obs size: {obs_with_z_flat.shape}")
        
#         reward_shaped = np.array([reward], dtype=np.float32)
#         done_shaped = [done]
        
#         return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions

#     def seed(self, seed):
#         if hasattr(self.env, 'seed'):
#             return self.env.seed(seed)

#     def close(self):
#         self.env.close()



# import numpy as np
# import copy
# from gym import spaces

# class AtariVMAPDWrapper(object):
#     def __init__(self, env, max_z, fix_z):
#         super().__init__()
#         self.env = env
#         self.max_z = max_z
#         self.fix_z = fix_z
#         self.cur_z = -1
#         self.num_agents = 1  # Atari is single agent
        
#         # DEBUG: Track skill usage
#         self.skill_usage_count = {i: 0 for i in range(max_z)}
#         self.total_resets = 0
        
#         # Create spaces for single agent (but keep list format for compatibility)
#         self.z_space = [spaces.Discrete(self.max_z)]
        
#         # IMPORTANT: Reset environment first to get actual observation shape
#         temp_obs = self.env.reset()
#         if isinstance(temp_obs, tuple):
#             actual_obs = temp_obs[0]
#         else:
#             actual_obs = temp_obs
        
#         # DEBUG: Print actual shapes
#         print(f"DEBUG Wrapper Init - fix_z: {fix_z}, max_z: {max_z}")
#         print(f"DEBUG: Original obs shape: {actual_obs.shape}")
#         print(f"DEBUG: Original obs size: {np.prod(actual_obs.shape)}")
        
#         # Calculate flattened size based on ACTUAL observation + z conditioning
#         z_vec = np.eye(self.max_z)[0]  # dummy z vector
#         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(actual_obs.shape[1], axis=1).repeat(actual_obs.shape[2], axis=2), actual_obs], axis=0)
#         flattened_size = obs_with_z.flatten().shape[0]
        
#         print(f"DEBUG: Natural flattened size: {flattened_size}")

#         # Now that discriminator handles z-stripping internally, use natural size
#         self.target_obs_size = flattened_size  # Your wrapper provides full size (101760)
        
#         print(f"DEBUG: Wrapper will provide: {flattened_size}")
#         print(f"DEBUG: Discriminator will strip {self.max_z} elements internally")
        
#         # CRITICAL FIX: Use the natural size since discriminator now handles z-stripping
#         self.observation_space = [spaces.Box(low=0, high=255, shape=(flattened_size,), dtype=np.uint8)]
#         self.share_observation_space = copy.deepcopy(self.observation_space)
#         self.action_space = [self.env.action_space]
        
#         # Z observation spaces (for DGPO)
#         self.z_obs_space = copy.deepcopy(self.observation_space)
#         self.z_local_obs_space = copy.deepcopy(self.observation_space)

#     def reset(self, fix_z=None):
#         # DEBUG: Track skill selection logic
#         old_z = self.cur_z
        
#         if fix_z is not None:
#             self.cur_z = fix_z
#             print(f"DEBUG Reset: Using provided fix_z={fix_z}")
#         elif self.fix_z is not None:
#             self.cur_z = self.fix_z
#             # print(f"DEBUG Reset: Using wrapper fix_z={self.fix_z}")
#         else:
#             self.cur_z = np.random.randint(self.max_z)
#             print(f"DEBUG Reset: Randomly selected z={self.cur_z}")
        
#         # Track usage
#         self.skill_usage_count[self.cur_z] += 1
#         self.total_resets += 1
        
#         # Print skill distribution every 100 resets
#         if self.total_resets % 100 == 0:
#             print(f"SKILL DISTRIBUTION after {self.total_resets} resets:")
#             for skill, count in self.skill_usage_count.items():
#                 pct = (count / self.total_resets) * 100
#                 print(f"  Skill {skill}: {count} times ({pct:.1f}%)")
        
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
        
#         # Provide the natural full observation with z-vector (101760)
#         # The discriminator will handle z-stripping internally
#         if len(obs_with_z_flat) > self.target_obs_size:
#             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
#         elif len(obs_with_z_flat) < self.target_obs_size:
#             # Pad with zeros if somehow too small
#             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
#             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
#         if self.total_resets % 500 == 0:  # Less frequent obs debug
#             print(f"DEBUG Reset: z={self.cur_z}, obs shape={obs_with_z_flat.shape}")
        
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

#         # Provide the natural full observation with z-vector (101760)
#         # The discriminator will handle z-stripping internally
#         if len(obs_with_z_flat) > self.target_obs_size:
#             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
#         elif len(obs_with_z_flat) < self.target_obs_size:
#             # Pad with zeros if somehow too small
#             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
#             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
#         reward_shaped = np.array([reward], dtype=np.float32)
#         done_shaped = [done]
        
#         return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions

#     def seed(self, seed):
#         if hasattr(self.env, 'seed'):
#             return self.env.seed(seed)

#     def close(self):
#         # Print final skill distribution
#         print(f"FINAL SKILL DISTRIBUTION after {self.total_resets} total resets:")
#         for skill, count in self.skill_usage_count.items():
#             pct = (count / self.total_resets) * 100 if self.total_resets > 0 else 0
#             print(f"  Skill {skill}: {count} times ({pct:.1f}%)")
#         self.env.close()



# #This version is for the debugging i did to fix the only using 1 of the 2 skills problme
# import numpy as np
# import copy
# from gym import spaces

# class AtariVMAPDWrapper(object):

# #Origonal trianing run code
#     # def __init__(self, env, max_z, fix_z):
#     #     super().__init__()
#     #     self.env = env
#     #     self.max_z = max_z
#     #     self.fix_z = fix_z
#     #     self.cur_z = -1
#     #     self.num_agents = 1
        
#     #     # DEBUG: Enhanced tracking
#     #     self.skill_usage_count = {i: 0 for i in range(max_z)}
#     #     self.skill_episodes_completed = {i: 0 for i in range(max_z)}
#     #     self.skill_total_rewards = {i: 0.0 for i in range(max_z)}
#     #     self.skill_step_counts = {i: 0 for i in range(max_z)}
#     #     self.total_resets = 0
#     #     self.episode_steps = 0
#     #     self.episode_reward = 0.0
        
#     #     # Create spaces for single agent
#     #     self.z_space = [spaces.Discrete(self.max_z)]
        
#     #     # Get observation shape
#     #     temp_obs = self.env.reset()
#     #     if isinstance(temp_obs, tuple):
#     #         actual_obs = temp_obs[0]
#     #     else:
#     #         actual_obs = temp_obs
        
#     #     print(f"🔧 DEBUG Wrapper Init - fix_z: {fix_z}, max_z: {max_z}")
        
#     #     # Calculate observation size with z-vector
#     #     z_vec = np.eye(self.max_z)[0]
#     #     obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(actual_obs.shape[1], axis=1).repeat(actual_obs.shape[2], axis=2), actual_obs], axis=0)
#     #     flattened_size = obs_with_z.flatten().shape[0]
        
#     #     self.target_obs_size = flattened_size
#     #     self.observation_space = [spaces.Box(low=0, high=255, shape=(flattened_size,), dtype=np.uint8)]
#     #     self.share_observation_space = copy.deepcopy(self.observation_space)
#     #     self.action_space = [self.env.action_space]
        
#     #     # Z observation spaces
#     #     self.z_obs_space = copy.deepcopy(self.observation_space)
#     #     self.z_local_obs_space = copy.deepcopy(self.observation_space)


#     #New to fix 1 skill only used issue
#     def __init__(self, env, max_z, fix_z):
#         super().__init__()
#         self.env = env
#         self.max_z = max_z
#         self.fix_z = fix_z
#         self.cur_z = -1
#         self.num_agents = 1
        
#         # DEBUG: Enhanced tracking
#         self.skill_usage_count = {i: 0 for i in range(max_z)}
#         self.skill_episodes_completed = {i: 0 for i in range(max_z)}
#         self.skill_total_rewards = {i: 0.0 for i in range(max_z)}
#         self.skill_step_counts = {i: 0 for i in range(max_z)}
#         self.total_resets = 0
#         self.episode_steps = 0
#         self.episode_reward = 0.0
        
#         # Create spaces for single agent
#         self.z_space = [spaces.Discrete(self.max_z)]
        
#         # Get observation shape
#         temp_obs = self.env.reset()
#         if isinstance(temp_obs, tuple):
#             actual_obs = temp_obs[0]
#         else:
#             actual_obs = temp_obs
        
#         print(f"🔧 DEBUG Wrapper Init - fix_z: {fix_z}, max_z: {max_z}")
#         print(f"🔧 Original obs shape: {actual_obs.shape}")
        
#         # 🔧 FIX: Simple concatenation size calculation
#         obs_flat_size = actual_obs.flatten().shape[0]
#         total_size = self.max_z + obs_flat_size  # z_vec + flattened_obs
        
#         print(f"🔧 Flattened obs size: {obs_flat_size}")
#         print(f"🔧 Z-vector size: {self.max_z}")
#         print(f"🔧 Total obs size: {total_size}")
        
#         # Create observation spaces with simple concatenation size
#         self.observation_space = [spaces.Box(low=0, high=255, shape=(total_size,), dtype=np.uint8)]
#         self.share_observation_space = copy.deepcopy(self.observation_space)
#         self.action_space = [self.env.action_space]
        
#         # Z observation spaces
#         self.z_obs_space = copy.deepcopy(self.observation_space)
#         self.z_local_obs_space = copy.deepcopy(self.observation_space)



# # #Origonal/modiifed versyion for pong
# #     def reset(self, fix_z=None):
# #         # Track episode completion BEFORE reset
# #         if self.episode_steps > 0:  # If this isn't the first reset
# #             self.skill_episodes_completed[self.cur_z] += 1
# #             self.skill_total_rewards[self.cur_z] += self.episode_reward
            
# #             # Log episode completion
# #             if self.skill_episodes_completed[self.cur_z] % 10 == 0:
# #                 avg_reward = self.skill_total_rewards[self.cur_z] / self.skill_episodes_completed[self.cur_z]
# #                 print(f"📊 SKILL {self.cur_z} Episode {self.skill_episodes_completed[self.cur_z]}: "
# #                       f"Steps={self.episode_steps}, Reward={self.episode_reward:.2f}, "
# #                       f"Avg={avg_reward:.2f}")
        
# #         # Reset episode tracking
# #         self.episode_steps = 0
# #         self.episode_reward = 0.0
        
# #         # Skill selection logic
# #         if fix_z is not None:
# #             self.cur_z = fix_z
# #             print(f"🎯 DEBUG Reset: Using provided fix_z={fix_z}")
# #         elif self.fix_z is not None:
# #             self.cur_z = self.fix_z
# #         else:
# #             self.cur_z = np.random.randint(self.max_z)
# #             print(f"🎯 DEBUG Reset: Randomly selected z={self.cur_z}")
        
# #         # Track usage
# #         self.skill_usage_count[self.cur_z] += 1
# #         self.total_resets += 1
        
# #         # Print comprehensive stats every 100 resets
# #         if self.total_resets % 100 == 0:
# #             print(f"\n📈 COMPREHENSIVE SKILL STATS after {self.total_resets} resets:")
# #             for skill in range(self.max_z):
# #                 episodes = self.skill_episodes_completed[skill]
# #                 avg_reward = self.skill_total_rewards[skill] / max(episodes, 1)
# #                 total_steps = self.skill_step_counts[skill]
# #                 usage_pct = (self.skill_usage_count[skill] / self.total_resets) * 100
                
# #                 print(f"  Skill {skill}: {self.skill_usage_count[skill]} resets ({usage_pct:.1f}%), "
# #                       f"{episodes} episodes, {total_steps} total steps, "
# #                       f"avg_reward={avg_reward:.2f}")
        
# #         # Reset environment
# #         result = self.env.reset()
# #         if isinstance(result, tuple):
# #             obs, info = result
# #         else:
# #             obs = result
        
# #         # Add z-vector to observation
# #         z_vec = np.eye(self.max_z)[self.cur_z]
# #         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
# #         # Create available actions
# #         num_actions = self.env.action_space.n
# #         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
# #         # Flatten and ensure correct size
# #         obs_with_z_flat = obs_with_z.flatten()
# #         if len(obs_with_z_flat) > self.target_obs_size:
# #             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
# #         elif len(obs_with_z_flat) < self.target_obs_size:
# #             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
# #             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
# #         return [obs_with_z_flat], [obs_with_z_flat], available_actions


# # #New for debugging:
# #     def reset(self, fix_z=None):
# #         # Track episode completion BEFORE reset
# #         if self.episode_steps > 0:
# #             self.skill_episodes_completed[self.cur_z] += 1
# #             self.skill_total_rewards[self.cur_z] += self.episode_reward
            
# #             if self.skill_episodes_completed[self.cur_z] % 10 == 0:
# #                 avg_reward = self.skill_total_rewards[self.cur_z] / self.skill_episodes_completed[self.cur_z]
# #                 print(f"📊 SKILL {self.cur_z} Episode {self.skill_episodes_completed[self.cur_z]}: "
# #                     f"Steps={self.episode_steps}, Reward={self.episode_reward:.2f}, "
# #                     f"Avg={avg_reward:.2f}")
        
# #         # Reset episode tracking
# #         self.episode_steps = 0
# #         self.episode_reward = 0.0
        
# #         # 🔍 DEBUG: Print what skill is being assigned
# #         old_z = self.cur_z
# #         print(f"🔧 RESET DEBUG: fix_z={fix_z}, self.fix_z={self.fix_z}, old_z={old_z}")
        
# #         # Skill selection logic
# #         if fix_z is not None:
# #             self.cur_z = fix_z
# #             print(f"🎯 RESET: Using provided fix_z={fix_z} → cur_z={self.cur_z}")
# #         elif self.fix_z is not None:
# #             self.cur_z = self.fix_z
# #             print(f"🎯 RESET: Using wrapper fix_z={self.fix_z} → cur_z={self.cur_z}")
# #         else:
# #             self.cur_z = np.random.randint(self.max_z)
# #             print(f"🎯 RESET: Randomly selected → cur_z={self.cur_z}")
        
# #         # Track usage
# #         self.skill_usage_count[self.cur_z] += 1
# #         self.total_resets += 1
        
# #         # Reset environment
# #         result = self.env.reset()
# #         if isinstance(result, tuple):
# #             obs, info = result
# #         else:
# #             obs = result
        
# #         # 🔍 DEBUG: Check z-vector creation
# #         z_vec = np.eye(self.max_z)[self.cur_z]
# #         print(f"🔧 Z-VECTOR DEBUG: cur_z={self.cur_z}, z_vec={z_vec}")
        
# #         # Add z-vector to observation
# #         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
# #         # Create available actions
# #         num_actions = self.env.action_space.n
# #         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
# #         # Flatten and ensure correct size
# #         obs_with_z_flat = obs_with_z.flatten()
# #         if len(obs_with_z_flat) > self.target_obs_size:
# #             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
# #         elif len(obs_with_z_flat) < self.target_obs_size:
# #             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
# #             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
# #         # 🔍 DEBUG: Check final observation
# #         final_z_check = obs_with_z_flat[:self.max_z]
# #         detected_skill = np.argmax(final_z_check) if np.sum(final_z_check) > 0 else -1
# #         print(f"🔧 FINAL OBS DEBUG: z_elements={final_z_check}, detected_skill={detected_skill}")
        
# #         if detected_skill != self.cur_z:
# #             print(f"⚠️  WARNING: Skill mismatch! Expected {self.cur_z}, got {detected_skill}")
        
# #         return [obs_with_z_flat], [obs_with_z_flat], available_actions
# # #stop


# #New to fix only using 1 still problem:
#     def reset(self, fix_z=None):
#         # Track episode completion BEFORE reset
#         if self.episode_steps > 0:
#             self.skill_episodes_completed[self.cur_z] += 1
#             self.skill_total_rewards[self.cur_z] += self.episode_reward
            
#             if self.skill_episodes_completed[self.cur_z] % 10 == 0:
#                 avg_reward = self.skill_total_rewards[self.cur_z] / self.skill_episodes_completed[self.cur_z]
#                 print(f"📊 SKILL {self.cur_z} Episode {self.skill_episodes_completed[self.cur_z]}: "
#                     f"Steps={self.episode_steps}, Reward={self.episode_reward:.2f}, "
#                     f"Avg={avg_reward:.2f}")
        
#         # Reset episode tracking
#         self.episode_steps = 0
#         self.episode_reward = 0.0
        
#         # 🔍 DEBUG: Print what skill is being assigned
#         old_z = self.cur_z
#         print(f"🔧 RESET DEBUG: fix_z={fix_z}, self.fix_z={self.fix_z}, old_z={old_z}")
        
#         # Skill selection logic
#         if fix_z is not None:
#             self.cur_z = fix_z
#             print(f"🎯 RESET: Using provided fix_z={fix_z} → cur_z={self.cur_z}")
#         elif self.fix_z is not None:
#             self.cur_z = self.fix_z
#             print(f"🎯 RESET: Using wrapper fix_z={self.fix_z} → cur_z={self.cur_z}")
#         else:
#             self.cur_z = np.random.randint(self.max_z)
#             print(f"🎯 RESET: Randomly selected → cur_z={self.cur_z}")
        
#         # Track usage
#         self.skill_usage_count[self.cur_z] += 1
#         self.total_resets += 1
        
#         # Reset environment
#         result = self.env.reset()
#         if isinstance(result, tuple):
#             obs, info = result
#         else:
#             obs = result
        
#         # 🔧 FIX: Use simple concatenation like original DGPO
#         z_vec = np.eye(self.max_z)[self.cur_z]
#         print(f"🔧 Z-VECTOR DEBUG: cur_z={self.cur_z}, z_vec={z_vec}")
        
#         # Flatten the observation first
#         obs_flat = obs.flatten()
        
#         # Simple concatenation: z_vec + flattened_obs
#         obs_with_z_flat = np.concatenate([z_vec, obs_flat])
        
#         # Update target size for simple concatenation
#         expected_size = self.max_z + obs_flat.shape[0]
        
#         # Create available actions
#         num_actions = self.env.action_space.n
#         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
#         # 🔍 DEBUG: Check final observation
#         final_z_check = obs_with_z_flat[:self.max_z]
#         detected_skill = np.argmax(final_z_check) if np.sum(final_z_check) > 0 else -1
#         print(f"🔧 FINAL OBS DEBUG: z_elements={final_z_check}, detected_skill={detected_skill}")
        
#         if detected_skill != self.cur_z:
#             print(f"⚠️  WARNING: Skill mismatch! Expected {self.cur_z}, got {detected_skill}")
#         else:
#             print(f"✅ SUCCESS: Skill match! Expected {self.cur_z}, got {detected_skill}")
        
#         return [obs_with_z_flat], [obs_with_z_flat], available_actions


# # #old for first training run:
# #     def step(self, actions):
# #         # Convert action
# #         if isinstance(actions, list):
# #             action = int(actions[0])
# #         else:
# #             action = int(actions)
        
# #         # Track steps
# #         self.episode_steps += 1
# #         self.skill_step_counts[self.cur_z] += 1
        
# #         # Step environment
# #         result = self.env.step(action)
# #         if len(result) == 4:
# #             obs, reward, done, info = result
# #         else:
# #             obs, reward, terminated, truncated, info = result
# #             done = terminated or truncated
        
# #         # Track reward
# #         self.episode_reward += reward
        
# #         # Log suspicious rewards or episodes
# #         if reward > 1.0 or reward < -1.0:  # Unusual Pong rewards
# #             print(f"🚨 UNUSUAL REWARD: Skill {self.cur_z}, Step {self.episode_steps}, Reward={reward}")
        
# #         if self.episode_steps > 200:  # Very long episode
# #             print(f"⏰ LONG EPISODE: Skill {self.cur_z}, {self.episode_steps} steps, reward={self.episode_reward:.2f}")
        
# #         # Structure info
# #         agent_info = dict(info)
# #         agent_info['bad_transition'] = False
# #         multi_agent_info = [agent_info]
        
# #         # Add z-vector to observation
# #         z_vec = np.eye(self.max_z)[self.cur_z]
# #         obs_with_z = np.concatenate([z_vec.reshape(-1, 1, 1).repeat(obs.shape[1], axis=1).repeat(obs.shape[2], axis=2), obs], axis=0)
        
# #         # Create available actions
# #         num_actions = self.env.action_space.n
# #         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
# #         # Flatten and ensure correct size
# #         obs_with_z_flat = obs_with_z.flatten()
# #         if len(obs_with_z_flat) > self.target_obs_size:
# #             obs_with_z_flat = obs_with_z_flat[:self.target_obs_size]
# #         elif len(obs_with_z_flat) < self.target_obs_size:
# #             padding = np.zeros(self.target_obs_size - len(obs_with_z_flat), dtype=obs_with_z_flat.dtype)
# #             obs_with_z_flat = np.concatenate([obs_with_z_flat, padding])
        
# #         reward_shaped = np.array([reward], dtype=np.float32)
# #         done_shaped = [done]
        
# #         return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions



# #New for fixing 1 skill only used issue:
#     def step(self, actions):
#         # Convert action
#         if isinstance(actions, list):
#             action = int(actions[0])
#         else:
#             action = int(actions)
        
#         # Track steps
#         self.episode_steps += 1
#         self.skill_step_counts[self.cur_z] += 1
        
#         # Step environment
#         result = self.env.step(action)
#         if len(result) == 4:
#             obs, reward, done, info = result
#         else:
#             obs, reward, terminated, truncated, info = result
#             done = terminated or truncated
        
#         # Track reward
#         self.episode_reward += reward
        
#         # Log suspicious rewards or episodes
#         if reward > 1.0 or reward < -1.0:  # Unusual Pong rewards
#             print(f"🚨 UNUSUAL REWARD: Skill {self.cur_z}, Step {self.episode_steps}, Reward={reward}")
        
#         if self.episode_steps > 200:  # Very long episode
#             print(f"⏰ LONG EPISODE: Skill {self.cur_z}, {self.episode_steps} steps, reward={self.episode_reward:.2f}")
        
#         # Structure info
#         agent_info = dict(info)
#         agent_info['bad_transition'] = False
#         multi_agent_info = [agent_info]
        
#         # 🔧 FIX: Use simple concatenation like original DGPO
#         z_vec = np.eye(self.max_z)[self.cur_z]
        
#         # Flatten the observation first
#         obs_flat = obs.flatten()
        
#         # Simple concatenation: z_vec + flattened_obs
#         obs_with_z_flat = np.concatenate([z_vec, obs_flat])
        
#         # Create available actions
#         num_actions = self.env.action_space.n
#         available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
#         reward_shaped = np.array([reward], dtype=np.float32)
#         done_shaped = [done]
        
#         return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions


#     def seed(self, seed):
#         if hasattr(self.env, 'seed'):
#             return self.env.seed(seed)

#     def close(self):
#         print(f"\n🏁 FINAL SKILL STATISTICS:")
#         print(f"Total resets: {self.total_resets}")
        
#         for skill in range(self.max_z):
#             episodes = self.skill_episodes_completed[skill]
#             avg_reward = self.skill_total_rewards[skill] / max(episodes, 1)
#             total_steps = self.skill_step_counts[skill]
#             usage_pct = (self.skill_usage_count[skill] / max(self.total_resets, 1)) * 100
            
#             print(f"  Skill {skill}:")
#             print(f"    Resets: {self.skill_usage_count[skill]} ({usage_pct:.1f}%)")
#             print(f"    Episodes completed: {episodes}")
#             print(f"    Total steps: {total_steps}")
#             print(f"    Total reward: {self.skill_total_rewards[skill]:.2f}")
#             print(f"    Average reward per episode: {avg_reward:.2f}")
            
#             if episodes == 0:
#                 print(f"    ⚠️  WARNING: Skill {skill} completed ZERO episodes!")
#             elif total_steps == 0:
#                 print(f"    ⚠️  WARNING: Skill {skill} took ZERO steps!")
        
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
        self.num_agents = 1
        
        # Episode tracking (needed for debugging if issues arise)
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Create spaces for single agent
        self.z_space = [spaces.Discrete(self.max_z)]
        
        # Get observation shape
        temp_obs = self.env.reset()
        if isinstance(temp_obs, tuple):
            actual_obs = temp_obs[0]
        else:
            actual_obs = temp_obs
        
        # Calculate observation size with simple concatenation
        obs_flat_size = actual_obs.flatten().shape[0]
        total_size = self.max_z + obs_flat_size  # z_vec + flattened_obs
        
        # Create observation spaces
        self.observation_space = [spaces.Box(low=0, high=255, shape=(total_size,), dtype=np.uint8)]
        self.share_observation_space = copy.deepcopy(self.observation_space)
        self.action_space = [self.env.action_space]
        
        # Z observation spaces (required for DGPO)
        self.z_obs_space = copy.deepcopy(self.observation_space)
        self.z_local_obs_space = copy.deepcopy(self.observation_space)

    def reset(self, fix_z=None):
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Skill selection logic
        if fix_z is not None:
            self.cur_z = fix_z
        elif self.fix_z is not None:
            self.cur_z = self.fix_z
        else:
            self.cur_z = np.random.randint(self.max_z)
        
        # Reset environment
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        # Simple concatenation: z_vec + flattened_obs
        z_vec = np.eye(self.max_z)[self.cur_z]
        obs_flat = obs.flatten()
        obs_with_z_flat = np.concatenate([z_vec, obs_flat])
        
        # Create available actions
        num_actions = self.env.action_space.n
        available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
        return [obs_with_z_flat], [obs_with_z_flat], available_actions

    def step(self, actions):
        # Convert action
        if isinstance(actions, list):
            action = int(actions[0])
        else:
            action = int(actions)
        
        # Track steps and reward (needed for episode completion)
        self.episode_steps += 1
        
        # Step environment
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        # Track reward
        self.episode_reward += reward
        
        # Structure info for multi-agent compatibility
        agent_info = dict(info)
        agent_info['bad_transition'] = False
        multi_agent_info = [agent_info]
        
        # Simple concatenation: z_vec + flattened_obs
        z_vec = np.eye(self.max_z)[self.cur_z]
        obs_flat = obs.flatten()
        obs_with_z_flat = np.concatenate([z_vec, obs_flat])
        
        # Create available actions
        num_actions = self.env.action_space.n
        available_actions = [np.ones((num_actions,), dtype=np.float32)]
        
        # reward_shaped = np.array([reward], dtype=np.float32)
        # done_shaped = [done]
        
        # return [obs_with_z_flat], [obs_with_z_flat], [reward_shaped], done_shaped, multi_agent_info, available_actions
        reward_reshaped = np.array([[reward]], dtype=np.float32)  # Shape: (1, 1)
        
        return [obs_with_z_flat], [obs_with_z_flat], reward_reshaped, [done], multi_agent_info, available_actions

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def close(self):
        self.env.close()