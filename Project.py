import numpy as np


class Expert:
    def __init__(self, id):
        self.id = id


class RegretMatchingDecisionMaker:

    def __init__(self, experts):
        self.n = len(experts)
        # Players
        self.experts = experts
        # cumulative expected reward for our Regret Matching algorithm
        self.expected_reward = 0.
        # cumulative expected rewards for Players
        self.experts_rewards = np.zeros(self.n)
        # cumulative regrets towards Players
        self.regrets = np.zeros(self.n)
        # probability disribution over Players to draw decision from
        self.p = np.full(self.n, 1. / self.n)

    def decision(self):
        expert = np.random.choice(self.experts, 1,  p=self.p)
        return expert[0]

    def update_rule(self, rewards_vector):
        self.expected_reward += np.dot(self.p, rewards_vector)
        self.experts_rewards += rewards_vector
        self.regrets = self.experts_rewards - self.expected_reward
        self._update_p()

    def _update_p(self):
        sum_w = np.sum([self._w(i) for i in np.arange(self.n)])
        if sum_w <= 0:
            self.p = np.full(self.n, 1. / self.n)
        else:
            self.p = np.asarray(
                [self._w(i) / sum_w for i in np.arange(self.n)]
            )

    def _w(self, i):
        return max(0, self.regrets[i])


ROCK = Expert('rock')
PAPER = Expert('paper')
SCISSORS = Expert('scissors')

RPS_EXPERTS = [ROCK, PAPER, SCISSORS]

RPS_REWARD_VECTORS = {
    ROCK:     np.asarray([0, 1, -1]),  
    PAPER:    np.asarray([-1, 0, 1]),  
    SCISSORS: np.asarray([1, -1, 0]),  
}


class RPSPlayer(RegretMatchingDecisionMaker):
    def __init__(self):
        super(RPSPlayer, self).__init__(RPS_EXPERTS)
        self.sum_p = np.full(3, 0.)
        self.games_played = 0

    def move(self):
        return self.decision()

    def learn_from(self, opponent_move):
        reward_vector = RPS_REWARD_VECTORS[opponent_move]
        self.update_rule(reward_vector)
        self.games_played += 1
        self.sum_p += self.p

    def current_best_response(self):
        return np.round(self.sum_p / self.games_played, 4)

    def eps(self):
        return np.max(self.regrets / self.games_played)
    def plot(self):
        pass


import numpy as np
import matplotlib.pyplot as plt



a = RPSPlayer()
b = RPSPlayer()

t = 10000
x=list()
y=list()

for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    
    a.learn_from(b_move)
    x.append(a.eps())
    b.learn_from(a_move)
    y.append(b.eps())

_2e = np.round(2 * np.max([a.eps(), b.eps()]), 3)
a_ne = a.current_best_response()
b_ne = b.current_best_response()
print("{0} - nash equilibrium for RPS: {1}, {2}".format(_2e, a_ne, b_ne))

ax = plt.subplot(111)
t = np.arange(0.0,10000, 1)

fig, ax = plt.subplots()

ax.plot(t,x,marker="o", label="Player 1")
ax.plot(t,y,marker="x", label="Player 2")

ax.set_ylabel('regret')
ax.set_xlabel('iteration')


ax.legend()

plt.legend()
plt.show()
print(a.p);
print(len(a.p))
