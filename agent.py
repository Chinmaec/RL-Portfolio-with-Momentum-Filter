import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head (output: raw portfolio weights)
        self.actor  = nn.Linear(hidden, action_dim)

        # Critic head (output: scalar- how good is this state?)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h      = self.shared(x)
        weights = self.actor(h)      
        value   = self.critic(h)     
        return weights, value


# PPO agent 
class PPOAgent:

    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=2):
        """
        lr        : learning rate
        gamma     : discount factor: gamma=0.99 -> reward 100 days later is worth 0.99^100 ? 37% now
        clip_eps  : PPO clip threshold (standard = 0.2)
        epochs    : how many gradient steps per batch of experience
        """
        self.gamma    = gamma
        self.clip_eps = clip_eps
        self.epochs   = epochs
        self._eps     = 1e-8

        self.net      = PolicyNetwork(state_dim, action_dim)
        self.opt      = optim.Adam(self.net.parameters(), lr=lr)
        self.action_std = 0.02

    #  Act
    def act(self, state, deterministic = False):
        """
        Given a state, return:
          - weights  : raw action (numpy)
          - log_prob : log probability of this action (needed for PPO update)
          - value    : critic's estimate of state value
        """
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)


        with torch.no_grad():
            weights, value = self.net(state_t)
        
        b = self.action_std / np.sqrt(2.0)  
        dist = torch.distributions.Laplace(weights, torch.full_like(weights, b))

        if deterministic:
            action = weights
            log_prob = 0.0  # not used in backtest
        else:

            action = dist.sample()
            log_prob = dist.log_prob(action).sum().item()

        return (
            action.squeeze().cpu().numpy(),
            log_prob,
            value.squeeze().item()
        )
        

    # Learn
    def learn(self, batch):
        """
        batch : list of (state, action, log_prob, reward, value) tuples

        Steps:
        1. Compute discounted returns G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        2. Compute advantage           A_t = G_t - V(s_t)
        3. PPO clip loss on actor
        4. MSE loss on critic
        5. Gradient step
        """
        rewards  = [b[3] for b in batch]

        states  = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        old_lps = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        values  = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)


        # 1. Discounted returns 
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # 2. Advantage
        
        # A_t = G_t - V(s_t)
        # Positive/ Negative advantage ? action was better/ worse than the baseline expected
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + self._eps)

        # 3+4. PPO Loss 
        for _ in range(self.epochs):
            weights_new, values_new = self.net(states)

            b = self.action_std / np.sqrt(2.0)
            dist = torch.distributions.Laplace(weights_new, torch.full_like(weights_new, b))
            new_lps  = dist.log_prob(actions).sum(dim=1)

            ratio    = torch.exp(new_lps - old_lps) # Probability ratio

            # Clipped surrogate objective
            surr1    = ratio * advantages
            surr2    = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(values_new.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  
            self.opt.step()

        return loss.item()

# Training Loop
def train(env, agent, n_episodes=200, batch_size=128):
    "Each episode - one full pass through the price history (train set)"
    episode_rewards = []

    for ep in range(n_episodes):
        state    = env.reset()
        batch    = []
        ep_reward = 0.0
        done     = False

        decision_state = None
        decision_action = None
        decision_log_prob = None
        decision_value = None
        holding_reward = 0.0
 
        while not done:

            if state is None: 
                break 
            
            if env.should_rebalance_today():
                if decision_state is not None:
                    batch.append((decision_state, decision_action, decision_log_prob, holding_reward, decision_value))
                    holding_reward = 0.0

                    if len(batch) >= batch_size:
                        agent.learn(batch)
                        batch = []

                # new decision
                decision_action, decision_log_prob, decision_value = agent.act(state)
                decision_state = state
                action_to_env = decision_action
            else: 
                action_to_env = None           

            next_state, reward, done = env.step(action_to_env)
            holding_reward += reward
            ep_reward += reward

            if not done:
                state = next_state

        if decision_state is not None:
            batch.append((decision_state, decision_action, decision_log_prob, holding_reward, decision_value))
        if batch:
            agent.learn(batch)

        episode_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {ep+1:>4} | Avg Reward (last 20): {avg:+.4f}")

    return episode_rewards
