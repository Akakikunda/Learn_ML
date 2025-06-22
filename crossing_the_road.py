import numpy as np
import random
from collections import defaultdict

class RoadCrossingEnv:
    def __init__(self):
        self.width = 10
        self.height = 5
        self.agent_pos = [0, self.height//2]  # Start at left side
        self.goal_pos = [self.width-1, self.height//2]  # Goal at right side
        self.traffic = self._generate_traffic()
        self.actions = ['left', 'right', 'forward']
        self.action_sequence = ['right', 'left', 'right']  # Target sequence
        self.current_step = 0
        self.done = False
        
    def _generate_traffic(self):
        # Generate moving obstacles (cars)
        traffic = []
        for lane in range(self.height):
            if lane != self.height//2:  # No traffic in agent's lane
                speed = random.choice([1, 2])
                direction = random.choice([-1, 1])
                traffic.append({'lane': lane, 'speed': speed, 'direction': direction, 'pos': 0})
        return traffic
    
    def reset(self):
        self.agent_pos = [0, self.height//2]
        self.traffic = self._generate_traffic()
        self.current_step = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        # Simplified state representation
        return tuple(self.agent_pos + [self.current_step % len(self.action_sequence)])
    
    def _move_traffic(self):
        for car in self.traffic:
            car['pos'] += car['direction'] * car['speed']
            if car['pos'] >= self.width or car['pos'] < 0:
                car['pos'] = 0 if car['direction'] > 0 else self.width-1
    
    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}
            
        # Execute action
        if action == 'left':
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 'right':
            self.agent_pos[1] = min(self.height-1, self.agent_pos[1] + 1)
        elif action == 'forward':
            self.agent_pos[0] += 1
        
        # Check if action matches desired sequence
        desired_action = self.action_sequence[self.current_step % len(self.action_sequence)]
        sequence_reward = 1 if action == desired_action else -0.5
        
        # Move traffic
        self._move_traffic()
        
        # Check for collisions
        collision = False
        for car in self.traffic:
            if self.agent_pos[1] == car['lane'] and abs(self.agent_pos[0] - car['pos']) < 1:
                collision = True
                break
        
        # Calculate reward
        if collision:
            reward = -10
            self.done = True
        elif self.agent_pos[0] >= self.width-1:
            reward = 10 + sequence_reward * 5  # Bonus for completing sequence
            self.done = True
        else:
            reward = sequence_reward
        
        self.current_step += 1
        return self._get_state(), reward, self.done, {}

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        return self.env.actions[np.argmax(self.q_table[state])]
    
    def update(self, state, action, reward, next_state, done):
        action_idx = self.env.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.lr * td_error
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train_agent(episodes=1000):
    env = RoadCrossingEnv()
    agent = QLearningAgent(env)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    return agent

# Train the agent
trained_agent = train_agent()

def test_agent(agent, num_tests=10):
    env = RoadCrossingEnv()
    successes = 0
    sequence_matches = 0
    
    for _ in range(num_tests):
        state = env.reset()
        actions_taken = []
        
        while True:
            action = agent.get_action(state)
            next_state, _, done, _ = env.step(action)
            actions_taken.append(action)
            state = next_state
            
            if done:
                if env.agent_pos[0] >= env.width-1:
                    successes += 1
                    # Check if sequence was followed at least once
                    for i in range(len(actions_taken)-2):
                        if actions_taken[i:i+3] == env.action_sequence:
                            sequence_matches += 1
                            break
                break
                
    print(f"Success rate: {successes/num_tests*100:.1f}%")
    print(f"Sequence match rate: {sequence_matches/num_tests*100:.1f}%")

test_agent(trained_agent)