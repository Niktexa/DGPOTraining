import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import cv2
import matplotlib
import matplotlib.pyplot as plt
from collections import deque

def _t2n(x):
    return x.detach().cpu().numpy()

class PongRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for Pong. See parent class for details."""
    def __init__(self, config):
        super(PongRunner, self).__init__(config)
        self.episode_rewards = []
        self.episode_count = 0
        self.total_episodes_completed = 0

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        self.episode_rewards = deque(maxlen=self.episode_length)

        for episode in range(episodes):
            self.episode_count = episode
            if self.episode_count % 5 == 0:
                print(f"Episode Number: {self.episode_count}")

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                # Sample actions
                ex_values, in_values, actions, action_log_probs, rnn_states, \
                    rnn_states_ex_critic, rnn_states_in_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                
                # Added to fix the intirinsic rewards being computed too late in the trianing: 
                z_log_probs, loc_z_log_probs, rnn_states_z, loc_rnn_states_z = self.VMAPD_collect(step)
                intrinsic_rewards = z_log_probs  # This is your diversity reward signal

                # Combine extrinsic and intrinsic rewards
                total_rewards = rewards + intrinsic_rewards

                # if episode % 5 == 0 and step == 0:
                #     print(f"\n=== EPISODE {episode} DEBUG ===")
                #     print(f"DEBUG BUFFER - Stored rewards: {total_rewards.flatten()}")


                # insert data into buffer
                data = dict()
                data['obs'] = obs
                data['share_obs'] = obs.copy()
                data['rnn_states_actor'] = rnn_states
                data['rnn_states_ex_critic'] = rnn_states_ex_critic
                data['rnn_states_in_critic'] = rnn_states_in_critic
                data['actions'] = actions
                data['action_log_probs'] = action_log_probs
                data['ex_value_preds'] = ex_values
                data['in_value_preds'] = in_values
                data['rewards'] = total_rewards
                data['dones'] = dones
                self.insert(data, step)

                # if episode % 5 == 0 and step == 0:
                #     print(f"DEBUG BUFFER - Stored rewards shape: {total_rewards.shape}")
                #     print(f"DEBUG BUFFER - Stored rewards: {total_rewards.flatten()}")
                #     print(f"DEBUG BUFFER - Intrinsic component: {intrinsic_rewards.flatten()}")
                    
                #     # Check what's in the buffer for intrinsic returns
                #     if hasattr(self.buffer, 'in_returns'):
                #         print(f"DEBUG BUFFER - In returns range: [{self.buffer.in_returns.min():.3f}, {self.buffer.in_returns.max():.3f}]")

                # SECOND insert - now for z-related data only (remove z_log_probs since it's already used)
                data = dict()
                data['rnn_states_z'] = rnn_states_z
                data['loc_rnn_states_z'] = loc_rnn_states_z
                data['z_log_probs'] = z_log_probs  # Keep this for your compute() method
                data['loc_z_log_probs'] = loc_z_log_probs
                data['dones'] = dones
                self.insert(data, step)




                # # ADD THE DEBUG HERE (immediately after the first insert):
                # if episode % 5 == 0 and step == 0:
                #     print(f"DEBUG BUFFER - Stored rewards shape: {rewards.shape}")
                #     print(f"DEBUG BUFFER - Stored rewards: {rewards.flatten()}")
                    
                #     # Check what's in the buffer for intrinsic returns
                #     if hasattr(self.buffer, 'in_returns'):
                #         print(f"DEBUG BUFFER - In returns range: [{self.buffer.in_returns.min():.3f}, {self.buffer.in_returns.max():.3f}]")


                # # VMAPD
                # z_log_probs, loc_z_log_probs, rnn_states_z, loc_rnn_states_z = self.VMAPD_collect(step)
                # data = dict()
                # data['rnn_states_z'] = rnn_states_z
                # data['loc_rnn_states_z'] = loc_rnn_states_z
                # data['z_log_probs'] = z_log_probs
                # data['loc_z_log_probs'] = loc_z_log_probs
                # data['dones'] = dones
                # self.insert(data, step)

                
                if infos is not None:
                    for info in infos:
                        if 'episode' in info[0].keys():
                            self.episode_rewards.append(info[0]['episode']['r'])
                            self.total_episodes_completed += 1

  

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            

            # if episode % 5 == 0:  # Log every 5 episodes
            #     end = time.time()
                
            #     print(f"\n{'='*60}")
            #     print(f"EPISODE {episode}/{episodes} | TIMESTEPS {total_num_steps}/{self.num_env_steps}")
            #     print(f"FPS: {int(total_num_steps / (end - start))}")


            #     if len(self.episode_rewards) > 0:
            #         recent_rewards = list(self.episode_rewards)[-min(10, len(self.episode_rewards)):]
            #         print(f"REWARDS (last {len(recent_rewards)} episodes):")
            #         print(f"  Mean: {np.mean(recent_rewards):.2f}")
            #         print(f"  Median: {np.median(recent_rewards):.2f}")
            #         print(f"  Min/Max: {np.min(recent_rewards):.2f}/{np.max(recent_rewards):.2f}")
            #         print(f"  Total episodes completed: {self.total_episodes_completed}")
            #     else:
            #         print("REWARDS: No completed episodes yet")
                
            #     # Training losses

            #     if train_infos:
            #         print(f"DETAILED LOSSES:")
            #         print(f"  Ex Value Loss: {train_infos.get('ex_value_loss', 'N/A'):.6f}")
            #         print(f"  In Value Loss: {train_infos.get('in_value_loss', 'N/A'):.6f}")
            #         print(f"  Z Loss: {train_infos.get('z_loss', 'N/A'):.6f}")
            #         print(f"  Policy Entropy: {train_infos.get('dist_entropy', 'N/A'):.6f}")  # Add this
                    
            #         # Check for intrinsic reward signals
            #         if 'cur_value_0' in train_infos and 'cur_value_1' in train_infos:
            #             print(f"  Value estimates - Ex: {train_infos['cur_value_0']:.3f}, In: {train_infos['cur_value_1']:.3f}")
                    
            #         if 'imp_weight' in train_infos:
            #             print(f"  Importance weight: {train_infos['imp_weight']:.6f}")
                    
                
            #     print(f"{'='*60}\n")

            #     if train_infos:
            #         print(f"DETAILED LOSSES:")
            #         print(f"  Ex Value Loss: {train_infos.get('ex_value_loss', 'N/A'):.6f}")
            #         print(f"  In Value Loss: {train_infos.get('in_value_loss', 'N/A'):.6f}")
            #         print(f"  Z Loss: {train_infos.get('z_loss', 'N/A'):.6f}")
                    
            #         # Check for intrinsic reward signals
            #         if 'cur_value_0' in train_infos and 'cur_value_1' in train_infos:
            #             print(f"  Value estimates - Ex: {train_infos['cur_value_0']:.3f}, In: {train_infos['cur_value_1']:.3f}")
                    
            #         if 'imp_weight' in train_infos:
            #             print(f"  Importance weight: {train_infos['imp_weight']:.6f}")

            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                # print("\n Game {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                #         .format(self.all_args.game_name,
                #                 self.algorithm_name,
                #                 self.experiment_name,
                #                 episode,
                #                 episodes,
                #                 total_num_steps,
                #                 self.num_env_steps,
                #                 int(total_num_steps / (end - start))))
                                
                if train_infos:
                    train_infos["FPS"] = int(total_num_steps / (end - start))
                    if len(self.episode_rewards) > 0:
                        train_infos["episode_rewards_mean"] = np.mean(self.episode_rewards)
                        train_infos["episode_rewards_median"] = np.median(self.episode_rewards)
                        train_infos["episode_rewards_min"] = np.min(self.episode_rewards)
                        train_infos["episode_rewards_max"] = np.max(self.episode_rewards)
                        # print(
                        #     "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        #         .format(
                        #             np.mean(self.episode_rewards),
                        #             np.median(self.episode_rewards), 
                        #             np.min(self.episode_rewards),
                        #             np.max(self.episode_rewards)
                        #         )
                        # )
                self.log_train(train_infos, total_num_steps)


            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        
        obs = self.envs.reset()
        share_obs = obs.copy()
        
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def VMAPD_collect(self, step):
        self.trainer.prep_rollout()
        z_log_prob, rnn_state_z = self.trainer.policy.evaluate_z(
            np.concatenate(self.buffer.share_obs[step+1]),
            np.concatenate(self.buffer.rnn_states_z[step]),
            np.concatenate(self.buffer.masks[step+1]),
            isTrain=False,
        )
        loc_z_log_prob, loc_rnn_state_z = self.trainer.policy.evaluate_local_z(
            np.concatenate(self.buffer.obs[step+1]),
            np.concatenate(self.buffer.loc_rnn_states_z[step]),
            np.concatenate(self.buffer.masks[step+1]),
            # isTrain=False,
        )

        # if step == 0:  # Only first step to avoid spam
        #     print(f"DEBUG VMAPD_collect:")
        #     print(f"  z_log_prob range: [{_t2n(z_log_prob).min():.3f}, {_t2n(z_log_prob).max():.3f}]")
        #     print(f"  loc_z_log_prob range: [{_t2n(loc_z_log_prob).min():.3f}, {_t2n(loc_z_log_prob).max():.3f}]")
        
        # [self.envs, agents, dim]
        z_log_probs = np.array(np.split(_t2n(z_log_prob), self.n_rollout_threads))
        rnn_states_z = np.array(np.split(_t2n(rnn_state_z), self.n_rollout_threads))
        loc_z_log_probs = np.array(np.split(_t2n(loc_z_log_prob), self.n_rollout_threads))
        loc_rnn_states_z = np.array(np.split(_t2n(loc_rnn_state_z), self.n_rollout_threads))

        return z_log_probs, loc_z_log_probs, rnn_states_z, loc_rnn_states_z

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        ex_value, in_value, action, action_log_prob, \
            rnn_states, rnn_states_ex_critic, rnn_states_in_critic \
                = self.trainer.policy.get_actions(
                    np.concatenate(self.buffer.share_obs[step]),
                    np.concatenate(self.buffer.obs[step]),
                    np.concatenate(self.buffer.rnn_states[step]),
                    np.concatenate(self.buffer.rnn_states_ex_critic[step]),
                    np.concatenate(self.buffer.rnn_states_in_critic[step]),
                    np.concatenate(self.buffer.masks[step])
                )
        # [self.envs, agents, dim]
        ex_values = np.array(np.split(_t2n(ex_value), self.n_rollout_threads))
        in_values = np.array(np.split(_t2n(in_value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_ex_critic = np.array(np.split(_t2n(rnn_states_ex_critic), self.n_rollout_threads))
        rnn_states_in_critic = np.array(np.split(_t2n(rnn_states_in_critic), self.n_rollout_threads))
        # rearrange action
        actions_env = actions

        # if self.episode_count % 5 == 0 and step == 0:
        #     print("DEBUG for collect method:")
        #     print(f"DEBUG COLLECT - Ex values range: [{ex_values.min():.3f}, {ex_values.max():.3f}]")
        #     print(f"DEBUG COLLECT - In values range: [{in_values.min():.3f}, {in_values.max():.3f}]")

        return ex_values, in_values, actions, action_log_probs, rnn_states,\
                    rnn_states_ex_critic, rnn_states_in_critic, actions_env

    def insert(self, data, step):    
        
        dones = (data['dones']==True)
        if 'rnn_states_actor' in data:
            data['rnn_states_actor'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'rnn_states_ex_critic' in data:
            data['rnn_states_ex_critic'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'rnn_states_in_critic' in data:
            data['rnn_states_in_critic'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'rnn_states_z' in data:
            data['rnn_states_z'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        if 'loc_rnn_states_z' in data:
            data['loc_rnn_states_z'][dones] = \
                np.zeros(((dones).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        data['masks'] = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        data['masks'][dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)
        
        self.buffer.insert(data, step)

    @torch.no_grad()
    def eval(self, total_num_steps):

        eval_episode_rewards = []

        seed_num = np.arange(self.n_eval_rollout_threads) // self.max_z 
        z_num = np.arange(self.n_eval_rollout_threads) % self.max_z

        eval_obs = self.eval_envs.seed(seed_num.astype('int'))
        eval_obs = self.eval_envs.reset(z_num)
        
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        finish_time_step = np.zeros(self.n_eval_rollout_threads)

        for eval_step in range(self.episode_length*10):

            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=False
            )

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            eval_actions_env = eval_actions

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards.flatten()*(finish_time_step==0))

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            finish_time_step += eval_dones.all(-1) * (finish_time_step==0) * eval_step

            if (finish_time_step>0).all():
                break

        eval_episode_rewards = np.array(eval_episode_rewards).sum(0)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.mean(eval_episode_rewards)
        eval_average_episode_rewards = eval_env_infos['eval_average_episode_rewards']
        eval_env_infos['eval_average_episode_length'] = np.mean(finish_time_step)
        eval_average_episode_length = eval_env_infos['eval_average_episode_length']
        print("eval average episode rewards {:.4f} eval_average_episode_length: {:.1f}"
            .format(
                eval_average_episode_rewards,
                eval_average_episode_length
            )
        )
        self.log_env(eval_env_infos, total_num_steps)


        # # Add this at the very end of the eval() method, after all existing code:
        # print(f"\nZ-BEHAVIOR TEST:")
        # for z in range(min(self.max_z, 3)):
        #     test_obs = self.eval_envs.reset([z])
        #     actions = []
        #     eval_rnn_states = np.zeros((1, self.recurrent_N, self.hidden_size), dtype=np.float32)
        #     eval_masks = np.ones((1, 1, 1), dtype=np.float32)
            
        #     for step in range(20):
        #         action, new_rnn_states = self.trainer.policy.act(
        #             np.concatenate(test_obs), 
        #             np.concatenate(eval_rnn_states),
        #             np.concatenate(eval_masks),
        #             deterministic=False
        #         )
                
        #         actions.append(int(_t2n(action)[0]))
                
        #         # Add state debugging every 5 steps
        #         if step % 5 == 0:
        #             print(f"    Step {step}: obs_shape={test_obs.shape}, obs_sample={test_obs.flatten()[:5]}")
                
        #         test_obs, _, _, _ = self.eval_envs.step([[_t2n(action)[0]]])
        #         eval_rnn_states = np.array(np.split(_t2n(new_rnn_states), 1))
            
        #     print(f"  Z={z}: actions {actions[:10]} (diversity: {len(set(actions))/len(actions):.2f})")
            
        #     # Try to get action probabilities with error handling
        #     try:
        #         with torch.no_grad():
        #             test_obs_reset = self.eval_envs.reset([z])
        #             test_rnn_states = np.zeros((1, self.recurrent_N, self.hidden_size), dtype=np.float32)
        #             test_masks = np.ones((1, 1, 1), dtype=np.float32)
                    
        #             actor_output = self.trainer.policy.actor(
        #                 np.concatenate(test_obs_reset),
        #                 np.concatenate(test_rnn_states),
        #                 np.concatenate(test_masks)
        #             )
                    
        #             # Handle different return formats
        #             if len(actor_output) >= 3:
        #                 dist = actor_output[-1]  # Distribution is usually last
        #                 probs = torch.softmax(dist.logits, dim=-1)
        #                 entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        #                 print(f"    Action probs: {_t2n(probs)[0]}")
        #                 print(f"    Policy entropy: {_t2n(entropy)[0]:.3f}")
        #     except Exception as e:
        #         print(f"    Could not get action probabilities: {e}")

    @torch.no_grad()
    def render(self):
        """Visualize the Pong environment."""
        
        envs = self.envs
        all_frames = []
        
        for z in range(self.max_z):
            self.envs.seed(self.seed)
            obs = envs.reset(z)

            if self.all_args.save_gifs:
                image = envs.render(mode='rgb_array')
                cv2.putText(image, f"Strategy {z}", (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255), 2)
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            episode_rewards = []
            
            for _ in range(self.episode_length):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                actions_env = actions   

                # Observe reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render(mode='rgb_array')
                    cv2.putText(image, f"Strategy {z}", (5, 25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255), 2)
                    all_frames.append(image)
                else:
                    envs.render('human')
                
                if dones.all():
                    break
            
            avg_rewards = np.sum(np.array(episode_rewards))
            # print(f"Strategy {z} average episode rewards: {avg_rewards}")

        if self.all_args.save_gifs:
            video_dir = str(self.gif_dir) + '/pong_render.avi'
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            if len(all_frames) > 0:
                h, w, c = all_frames[0].shape
                gout = cv2.VideoWriter(video_dir, fourcc, 30.0, (w, h), True)
                for frame in all_frames:
                    gout.write(frame)
                gout.release()
                # print(f"Video saved to: {video_dir}")