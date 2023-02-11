from our_implementation.net import ActorCriticNetwork
from torch.optim import Adam

class PPO:
    def __init__(self, env, lr):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.network = ActorCriticNetwork(shared_dims=[self.obs_dim, 512, 512], policy_dims=[256, 128, self.act_dim], value_dims=[256, 128, 1])

        self.optim = Adam(self.network.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        #self.logger = {
        #    'delta_t': time.time_ns(),
        #    't_so_far': 0,  # timesteps so far
        #    'i_so_far': 0,  # iterations so far
        #    'batch_lens': [],  # episodic lengths in batch
        #    'batch_rews': [],  # episodic returns in batch
        #    'actor_losses': [],  # losses of actor network in current iteration
        #}

    def learn(self):
