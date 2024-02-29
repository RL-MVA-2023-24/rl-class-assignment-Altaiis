from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import joblib

PATH = "bestmodel.joblib"

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# 10BN score



class ProjectAgent:

    def __init__(self):
        self.gamma = 0.9
        self.Q = None
        self.env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.nb_steps = 17
        self.nb_iterations = 30
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
    
    def greedy_action(self,s):
        Qsa = []
        nb_actions = self.env.action_space.n
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)


    def collect_samples(self, horizon, randomness=0.0, print_done_states=False):
        s, _ = self.env.reset()
        for _ in range(horizon):
            if np.random.rand() < randomness:
                a = self.greedy_action(s)
            else:
                a = self.env.action_space.sample()

            s2, r, done, trunc, _ = self.env.step(a)
            self.S.append(s)
            self.A.append(a)
            self.R.append(r)
            self.S2.append(s2)
            self.D.append(done)
            
            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
    

    # return the list of Q functions
    def rf_fqi(self, iterations, start=False):
        nb_actions = self.env.action_space.n
        nb_samples = len(self.S)

        S = np.array(self.S)
        A = np.array(self.A).reshape((-1,1))
        SA = np.append(S,A,axis=1)
        Q = self.Q

        for iter in range(iterations):
            if iter==0 and Q is None:
                value=self.R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((len(self.S),1))
                    S2A2 = np.append(self.S2,A2,axis=1)
                    Q2[:,a2] = Q.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = np.array(self.R) + self.gamma*(1-np.array(self.D))*max_Q2
            
            Q = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
            Q.fit(SA,value)
        self.Q = Q

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.env.action_space.n)
        else:
            return self.greedy_action(observation)

    def train(self, horizon):
        self.collect_samples(horizon)
        self.rf_fqi(self.nb_iterations,start=True)
        #print(0, evaluate_HIV(agent=self, nb_episode=5) / 1000000)
        print(0)


        for _ in range(self.nb_steps):
            self.collect_samples(500, randomness=0.85)
            self.rf_fqi(self.nb_iterations)
            #print(_+1, evaluate_HIV(agent=self, nb_episode=5) / 1000000)
            print(_+1)

            if _ == 10 or _ == 15 :
                print(_+1, evaluate_HIV(agent=self, nb_episode=5) / 1000000)
                self.save(PATH)


        self.save(PATH)
        print(_+1, evaluate_HIV(agent=self, nb_episode=5) / 1000000)

    def save(self, path):
        joblib.dump(self.Q, 'bestmodel.joblib')

    def load(self):
        self.Q = joblib.load('bestmodel.joblib')

if __name__ == "__main__":
    horizon = 2000

    # The time wrapper limits the number of steps in an episode at 200.
    # Now is the floor is yours to implement the agent and train it.

    # train the agent
    agent = ProjectAgent()
    agent.train(horizon)