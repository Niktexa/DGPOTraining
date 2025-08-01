# """
# TensorBoard Logger for DGPO Training
# Logs all DGPO-specific metrics to TensorBoard for real-time monitoring
# """

# import os
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# class DGPOTensorBoardLogger:
#     def __init__(self, log_dir, experiment_name):
#         """
#         Initialize TensorBoard logger for DGPO training
        
#         Args:
#             log_dir: Base directory for logs
#             experiment_name: Name of the experiment
#         """
#         self.log_dir = os.path.join(log_dir, "tensorboard", experiment_name)
#         self.writer = SummaryWriter(log_dir=self.log_dir)
#         self.episode_count = 0
        
#         print(f"🔥 TensorBoard logging to: {self.log_dir}")
#         print(f"📊 View dashboard: tensorboard --logdir {self.log_dir}")
        
#     def log_training_metrics(self, train_info, episode):
#         """
#         Log DGPO training metrics to TensorBoard
        
#         Args:
#             train_info: Dictionary of training metrics from DGPO
#             episode: Current episode number
#         """
#         self.episode_count = episode
        
#         # DGPO Core Metrics
#         if 'z_loss' in train_info:
#             self.writer.add_scalar('DGPO/z_loss', train_info['z_loss'], episode)
            
#         if 'diver_mask' in train_info:
#             self.writer.add_scalar('DGPO/diver_mask', train_info['diver_mask'], episode)
            
#         if 'Rex_mask' in train_info:
#             self.writer.add_scalar('DGPO/Rex_mask', train_info['Rex_mask'], episode)
            
#         # Skill-specific values
#         for i in range(10):  # Support up to 10 skills
#             skill_key = f'cur_value_{i}'
#             if skill_key in train_info:
#                 self.writer.add_scalar(f'DGPO/Skills/cur_value_{i}', train_info[skill_key], episode)
        
#         # Training Performance
#         if 'policy_loss' in train_info:
#             self.writer.add_scalar('Training/policy_loss', train_info['policy_loss'], episode)
            
#         if 'ex_value_loss' in train_info:
#             self.writer.add_scalar('Training/ex_value_loss', train_info['ex_value_loss'], episode)
            
#         if 'in_value_loss' in train_info:
#             self.writer.add_scalar('Training/in_value_loss', train_info['in_value_loss'], episode)
            
#         if 'dist_entropy' in train_info:
#             self.writer.add_scalar('Training/dist_entropy', train_info['dist_entropy'], episode)
            
#         if 'imp_weight' in train_info:
#             self.writer.add_scalar('Training/importance_weights', train_info['imp_weight'], episode)
            
#     def log_episode_metrics(self, episode_rewards, episode_length, episode):
#         """
#         Log episode-level metrics
        
#         Args:
#             episode_rewards: Rewards from the episode
#             episode_length: Length of the episode
#             episode: Episode number
#         """
#         if isinstance(episode_rewards, (list, np.ndarray)):
#             avg_reward = np.mean(episode_rewards)
#             total_reward = np.sum(episode_rewards)
#         else:
#             avg_reward = episode_rewards
#             total_reward = episode_rewards
            
#         self.writer.add_scalar('Environment/average_step_reward', avg_reward, episode)
#         self.writer.add_scalar('Environment/total_episode_reward', total_reward, episode)
#         self.writer.add_scalar('Environment/episode_length', episode_length, episode)
        
#     def log_skill_discovery_progress(self, z_log_probs, episode):
#         """
#         Log skill discovery specific metrics
        
#         Args:
#             z_log_probs: Discriminator outputs for skill identification
#             episode: Episode number
#         """
#         if z_log_probs is not None:
#             # Convert to numpy if tensor
#             if hasattr(z_log_probs, 'detach'):
#                 z_log_probs = z_log_probs.detach().cpu().numpy()
                
#             # Log discriminator confidence
#             max_confidence = np.max(z_log_probs)
#             mean_confidence = np.mean(z_log_probs)
            
#             self.writer.add_scalar('SkillDiscovery/max_discriminator_confidence', max_confidence, episode)
#             self.writer.add_scalar('SkillDiscovery/mean_discriminator_confidence', mean_confidence, episode)
            
#     def log_stage_transitions(self, stage_info, episode):
#         """
#         Log DGPO stage transition information
        
#         Args:
#             stage_info: Dict with stage information
#             episode: Episode number
#         """
#         if 'current_stage' in stage_info:
#             # 1 = Diversity Learning, 2 = Performance Learning
#             self.writer.add_scalar('DGPO/training_stage', stage_info['current_stage'], episode)
            
#         if 'diversity_threshold' in stage_info:
#             self.writer.add_scalar('DGPO/diversity_threshold', stage_info['diversity_threshold'], episode)
            
#         if 'performance_threshold' in stage_info:
#             self.writer.add_scalar('DGPO/performance_threshold', stage_info['performance_threshold'], episode)
            
#     def log_hyperparameters(self, hparams):
#         """
#         Log hyperparameters for the experiment
        
#         Args:
#             hparams: Dictionary of hyperparameters
#         """
#         # Log hyperparameters to TensorBoard
#         self.writer.add_hparams(
#             hparams,
#             {'final_episode': 0}  # Will be updated at end of training
#         )
        
#     def close(self):
#         """Close the TensorBoard writer"""
#         if self.writer:
#             self.writer.close()
#             print(f"📊 TensorBoard logs saved to: {self.log_dir}")
#             print(f"🔍 View results: tensorboard --logdir {self.log_dir}")


# # Convenience function for integration
# def create_dgpo_logger(run_dir, experiment_name):
#     """
#     Create a DGPO TensorBoard logger
    
#     Args:
#         run_dir: Base run directory
#         experiment_name: Name of the experiment
        
#     Returns:
#         DGPOTensorBoardLogger instance
#     """
#     return DGPOTensorBoardLogger(run_dir, experiment_name)



import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class DGPOTensorBoardLogger:
    """TensorBoard logger specifically designed for DGPO training metrics"""
    
    def __init__(self, log_dir, experiment_name, max_z=10):
        """
        Initialize the DGPO TensorBoard logger
        
        Args:
            log_dir (str): Directory to save TensorBoard logs
            experiment_name (str): Name of the experiment
            max_z (int): Maximum number of skills/latent variables
        """
        self.log_dir = os.path.join(log_dir, "tensorboard")
        self.max_z = max_z
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"🔥 TensorBoard logging to: {self.log_dir}")
        print(f"📊 View dashboard: tensorboard --logdir {self.log_dir}")
        print(f"🌐 Then open: http://localhost:6006")
        
    def log_training_metrics(self, train_info, episode):
        """
        Log DGPO training metrics to TensorBoard
        
        Args:
            train_info: Dictionary of training metrics from DGPO
            episode: Current episode number
        """
        # DGPO Core Metrics
        if 'z_loss' in train_info:
            self.writer.add_scalar('DGPO/z_loss', train_info['z_loss'], episode)
            
        if 'diver_mask' in train_info:
            self.writer.add_scalar('DGPO/diver_mask', train_info['diver_mask'], episode)
            
        if 'Rex_mask' in train_info:
            self.writer.add_scalar('DGPO/Rex_mask', train_info['Rex_mask'], episode)
            
        # Skill-specific values - use actual max_z instead of hardcoded 10
        for i in range(self.max_z):
            skill_key = f'cur_value_{i}'
            if skill_key in train_info:
                self.writer.add_scalar(f'DGPO/Skills/cur_value_{i}', train_info[skill_key], episode)
        
        # Training Performance
        if 'policy_loss' in train_info:
            self.writer.add_scalar('Training/policy_loss', train_info['policy_loss'], episode)
            
        if 'ex_value_loss' in train_info:
            self.writer.add_scalar('Training/ex_value_loss', train_info['ex_value_loss'], episode)
            
        if 'in_value_loss' in train_info:
            self.writer.add_scalar('Training/in_value_loss', train_info['in_value_loss'], episode)
            
        if 'dist_entropy' in train_info:
            self.writer.add_scalar('Training/dist_entropy', train_info['dist_entropy'], episode)
            
        if 'imp_weight' in train_info:
            self.writer.add_scalar('Training/importance_weights', train_info['imp_weight'], episode)
            
    def log_episode_metrics(self, episode_rewards, episode_length, episode):
        """
        Log episode-level metrics
        
        Args:
            episode_rewards: Rewards from the episode
            episode_length: Length of the episode
            episode: Episode number
        """
        if isinstance(episode_rewards, (list, np.ndarray)):
            avg_reward = np.mean(episode_rewards)
            total_reward = np.sum(episode_rewards)
        else:
            avg_reward = episode_rewards
            total_reward = episode_rewards
            
        self.writer.add_scalar('Environment/average_step_reward', avg_reward, episode)
        self.writer.add_scalar('Environment/total_episode_reward', total_reward, episode)
        self.writer.add_scalar('Environment/episode_length', episode_length, episode)
        
    def log_skill_discovery_progress(self, z_log_probs, episode):
        """
        Log skill discovery specific metrics
        
        Args:
            z_log_probs: Discriminator outputs for skill identification
            episode: Episode number
        """
        if z_log_probs is not None:
            # Convert to numpy if tensor
            if hasattr(z_log_probs, 'detach'):
                z_log_probs = z_log_probs.detach().cpu().numpy()
                
            # Log discriminator confidence
            max_confidence = np.max(z_log_probs)
            mean_confidence = np.mean(z_log_probs)
            
            self.writer.add_scalar('SkillDiscovery/max_discriminator_confidence', max_confidence, episode)
            self.writer.add_scalar('SkillDiscovery/mean_discriminator_confidence', mean_confidence, episode)
            
    def log_stage_transitions(self, stage_info, episode):
        """
        Log DGPO stage transition information
        
        Args:
            stage_info: Dict with stage information
            episode: Episode number
        """
        if 'current_stage' in stage_info:
            # 1 = Diversity Learning, 2 = Performance Learning
            self.writer.add_scalar('DGPO/training_stage', stage_info['current_stage'], episode)
            
        if 'diversity_threshold' in stage_info:
            self.writer.add_scalar('DGPO/diversity_threshold', stage_info['diversity_threshold'], episode)
            
        if 'performance_threshold' in stage_info:
            self.writer.add_scalar('DGPO/performance_threshold', stage_info['performance_threshold'], episode)
            
    def log_hyperparameters(self, hparams):
        """
        Log hyperparameters for the experiment
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        # Filter out 'unknown' values for cleaner display
        clean_hparams = {k: v for k, v in hparams.items() if v != 'unknown'}
        
        # Log hyperparameters to TensorBoard
        self.writer.add_hparams(
            clean_hparams,
            {'final_episode': 0}  # Will be updated at end of training
        )
        
    def close(self):
        """Close the TensorBoard writer"""
        if self.writer:
            self.writer.close()
            print(f"\n📊 TensorBoard logs saved to: {self.log_dir}")
            print(f"🔍 View results: tensorboard --logdir {self.log_dir}")
            print(f"🌐 Dashboard: http://localhost:6006")


# Convenience function for integration
def create_dgpo_logger(run_dir, experiment_name, max_z=10):
    """
    Create a DGPO TensorBoard logger
    
    Args:
        run_dir: Base run directory
        experiment_name: Name of the experiment
        max_z: Maximum number of skills
        
    Returns:
        DGPOTensorBoardLogger instance
    """
    return DGPOTensorBoardLogger(run_dir, experiment_name, max_z)