import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99
lambda_gae = 0.95
clip_epsilon = 0.2
update_steps = 10
epochs = 10
mini_batch_size = 64

# Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Actor (policy network)
        self.actor = nn.Linear(64, action_dim)
        # Critic (value network)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_action(self, x):
        x = self.forward(x)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, x, actions):
        x = self.forward(x)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(x)
        return action_log_probs, torch.squeeze(value), entropy

    def get_value(self, x):
        x = self.forward(x)
        return self.critic(x)

    

# Rollout Buffer
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        del self.advantages[:]
        del self.returns[:]

# PPO Agent
class PPOAgent:
    def __init__(self, input_dim, action_dim):
        self.policy = ActorCritic(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer()
        self.load_model()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, log_prob = self.policy.get_action(state)
        value = self.policy.get_value(state)
        return action.item(), log_prob, value

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * lambda_gae * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self):
        states = torch.FloatTensor(self.buffer.states).to(device)
        actions = torch.LongTensor(self.buffer.actions).to(device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(device)
        returns = torch.FloatTensor(self.buffer.returns).to(device)
        advantages = torch.FloatTensor(self.buffer.advantages).to(device)

        for _ in range(epochs):
            for idx in range(0, len(states), mini_batch_size):
                batch_states = states[idx:idx+mini_batch_size]
                batch_actions = actions[idx:idx+mini_batch_size]
                batch_old_log_probs = old_log_probs[idx:idx+mini_batch_size]
                batch_returns = returns[idx:idx+mini_batch_size]
                batch_advantages = advantages[idx:idx+mini_batch_size]

                # Get new action log_probs, state values and entropy
                log_probs, state_values, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                state_values = torch.squeeze(state_values)

                # Ratio for clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = (batch_returns - state_values).pow(2).mean()

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                # Update the policy network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(value)

    def finish_path(self, next_value):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]
        self.buffer.advantages = advantages
        self.buffer.returns = returns

    def save_model(self):
        # Assuming 'model' is an instance of a torch.nn.Module class
        torch.save(self.policy.state_dict(), "model_weights.pt", weights_only=True)

    def load_model(self):
        try:
            self.policy.load_state_dict(torch.load("model_weights.pt"))
            #self.policy.eval()
        except Exception as e:
            print(e)


# Main Training Loop
if __name__ == "__main__":
    env = gym.make('CartPole-v1')#, render_mode="human")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(obs_dim, action_dim)

    max_episodes = 1000
    max_timesteps = 500
    checkpoint_steps = 100

    for episode in range(max_episodes):
        state = env.reset()[0]
        total_reward = 0

        for t in range(max_timesteps):
            #if(t % checkpoint_steps == 0):
            #    agent.save_model()

            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.store_transition(state, action, log_prob, reward, done, value)
            total_reward += reward
            state = next_state

            if done or t == max_timesteps - 1:
                next_value = agent.policy.get_value(torch.FloatTensor(next_state).to(device)).item()
                agent.finish_path(next_value)
                agent.update()
                agent.buffer.clear()
                break

        print(f'Episode: {episode+1}, Total Reward: {total_reward}')

    env.close()
