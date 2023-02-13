from our_implementation.net import ActorCriticNetwork
from torch.optim import Adam
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from torch.nn import MSELoss
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from gym import spaces
from typing import Optional
import time
from stable_baselines3.common.utils import safe_mean



'''
   Disclaimer: We followed the tutorial at https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
   for this implementation with significant simplifications and adaptations for our requirements
   (such as implementing an ActorCriticNetwork with shared layers for policy and value function networks).
   No code was copied.
   
   Also we inherit from stable-baselines3 BaseAlgorithm in order to integrate our implementation with the simulation code
   which relies on many stable-baselines built-in fuctions.
'''


class OurPPO(BaseAlgorithm):
    def _setup_model(self) -> None:  # this is an abstract method from BaseAlgorithm
        self._setup_lr_schedule()
        #self.set_random_seed(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def __init__(self, policy, env, lr=3e-4, ts_per_batch=1210, max_ts_per_episode=1000000, n_updates_per_iter=5, gamma=0.99,
                 expl_std=0.5, tensorboard_log=None, use_stable_bl_policy=True, device='cpu',
                 _init_setup_model=True, clip=0.2, seed=0):
        super(OurPPO, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=lr,
            policy_kwargs={},
            verbose=1,
            device=device,
            create_eval_env=False,
            support_multi_env=False,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=(
                spaces.Box,
            ),
        )

        if _init_setup_model:
            self._setup_model()

        self.env = env
        if env is not None:
            self.obs_dim = env.observation_space.shape[0]
            self.act_dim = env.action_space.shape[0]

        self.use_stable_bl_policy = use_stable_bl_policy
        if not use_stable_bl_policy:
            self.network = ActorCriticNetwork(shared_dims=[self.obs_dim, 512], policy_dims=[256, 128, self.act_dim], value_dims=[256, 128, 1])
            self.optim = Adam(self.network.parameters(), lr=lr)

        # Set hyperparameters
        self.ts_per_batch = ts_per_batch
        self.max_ts_per_episode = max_ts_per_episode
        self.n_updates_per_iter = n_updates_per_iter
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.max_grad_norm = 0.5
        self.exploration_std = expl_std
        # self.save_freq = 10  # how often we save the model in iterations
        self.vf_coef = 0.5  # how the value loss is weighted in relation to the policy loss
        self.seed = seed
        torch.manual_seed(self.seed)
        print(f"PPO Torch seed: {self.seed}")

    def learn_custom(self, timesteps, log_interval, callback):
        total_timesteps, callback = self._setup_learn(
            timesteps, None, callback, -1, 5, None, True, "OurPPOAlgorithm"
        )

        n_rollouts = 0
        while self.num_timesteps < timesteps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            n_rollouts += 1
            print('Rollout number: ' + str(n_rollouts))
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(callback)
            self.num_timesteps += np.sum(batch_lens)
            print(f'Total timesteps: {self.num_timesteps}/{timesteps}')

            # Display training infos
            if log_interval is not None and n_rollouts % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", n_rollouts, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            # --- here, train() is called in OnPolicyAlgorithm
            pg_losses, value_losses = [],[]
            if not self.use_stable_bl_policy:
                V_phi_k = self.network.forward(batch_obs, calculate_policy=False).squeeze()
            else:
                V_phi_k, _, _ = self.policy.evaluate_actions(batch_obs, batch_acts)
                V_phi_k = V_phi_k.flatten()
            A_k = batch_rtgs - V_phi_k.detach()  # advantage
            A_k_normalized = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # according to the tutorial this is necessary

            for _ in range(self.n_updates_per_iter):
                # calculate log probs of rolled out actions with current parameters
                if not self.use_stable_bl_policy:
                    mean_action = self.network(batch_obs)
                    dist = MultivariateNormal(mean_action, covariance_matrix=self.exploration_std * torch.eye(self.act_dim))
                    log_probs = dist.log_prob(batch_acts)
                    V_phi = self.network.forward(batch_obs, calculate_policy=False).squeeze()
                else:
                    V_phi, log_probs, _ = self.policy.evaluate_actions(batch_obs, batch_acts)
                    V_phi = V_phi.flatten()

                policy_ratios = torch.exp(log_probs - batch_log_probs)

                # Calculate PPO objective
                part1 = policy_ratios * A_k_normalized
                part2 = torch.clamp(policy_ratios, 1 - self.clip, 1 + self.clip) * A_k_normalized
                pg_loss = (-torch.min(part1, part2)).mean()
                pg_losses.append(pg_loss.item())

                # Calculate value loss
                vf_loss = MSELoss()(V_phi.squeeze(), batch_rtgs.squeeze())
                value_losses.append(vf_loss.item())

                loss = pg_loss + self.vf_coef * vf_loss

                if not self.use_stable_bl_policy:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                else:
                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            self._n_updates += self.n_updates_per_iter
            #explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            # Logs
            #self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            #self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            #self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            #self.logger.record("train/explained_variance", explained_var)
            #if hasattr(self.policy, "log_std"):
            #    self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            #self.logger.record("train/clip_range", clip_range)
            #if self.clip_range_vf is not None:
            #    self.logger.record("train/clip_range_vf", clip_range_vf)
        return self

    def rollout(self, callback):
        # This function samples a dataset on-policy with self.ts_per_batch timesteps

        # Initilaize
        t = 0  # counts total timesteps per batch, not per episode
        batch_obs = []
        batch_acts = []
        batch_log_probs = []  # log probs of actions
        batch_rews = []  # batch rewards
        batch_lens = []  # episodic lengths

        while t < self.ts_per_batch:
            ep_rews = []
            obs = self.env.reset()
            for ep_t in range(self.max_ts_per_episode):
                t += 1

                # roll out 1 timestep
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
                callback.on_step()
                # done = truncated or terminated

                # Save in lists
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break  # the inner loop, so env will be reset

            batch_lens.append(ep_t + 1)
            batch_rews.append(np.array(ep_rews))

        batch_rtgs = self.compute_rtgs(batch_rews)  # Computed discounted rewards to go

        # return the dataset for this rollout
        return torch.tensor(np.array(batch_obs), dtype=torch.float32),\
               torch.tensor(np.array(batch_acts), dtype=torch.float32),\
               torch.tensor(np.array(batch_log_probs), dtype=torch.float32),\
               batch_rtgs, batch_lens

    def get_action(self, obs):
        # here we sample actions with gaussian noise (for exploration)
        # from our model's prediction and calculate the log prob
        if not self.use_stable_bl_policy:
            mean_action = self.network(obs)
            dist = MultivariateNormal(mean_action, covariance_matrix=self.exploration_std*torch.eye(self.act_dim))
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            actions, _, log_prob = self.policy(torch.tensor(obs, dtype=torch.float32))
            action = actions.flatten()
            #log_prob = log_probs.item()

        # don't need computation graphs for log prob and action, so we call detach()
        return action.detach().numpy(), log_prob.detach().numpy()

    def compute_rtgs(self, batch_rews):
        # input: batch_rews is a list of lists with rewards collected from the environment
        # we want to calculate discounted rewards for each list (which represents one episode)

        batch_rtgs_ls = []  # will contain rtgs in reverse order
        for ep_rews in reversed(batch_rews):  # we reverse both lists to efficiently calculate discounted rewards
            discounted_reward = 0  # for this episode so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs_ls.append(discounted_reward)

        return torch.tensor(np.array(batch_rtgs_ls[::-1]), dtype=torch.float32)


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        return self.learn_custom(timesteps=total_timesteps, log_interval=log_interval, callback=callback)


#import gym
#env = gym.make('Pendulum-v1')
#model = PPO(env, lr=0.005, gamma=0.95, ts_per_batch=4800, max_ts_per_episode=1600, n_updates_per_iter=5, expl_std=0.5)
#model.learn(10000)