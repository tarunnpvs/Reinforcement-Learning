import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random



class DQNSolver():
  def __init__(self,env,nstate,nact,alpha=0.01,epsilon=1,alpha_decay=0.01,epsilon_decay=0.95,gamma=1,len=1000000):
    self.env = env
    self.nstate = nstate
    self.nact = nact
    self.alpha = alpha
    self.epsilon = epsilon
    self.alpha_decay = alpha_decay
    self.epsilon_decay = epsilon_decay
    self.gamma = gamma
    self.memory = deque(maxlen = len)
    self.x = tf.placeholder(shape = [None,nstate],dtype = tf.float32,name = "in")
    self.y = tf.placeholder(shape =[None,nact],dtype = tf.float32,name = "out" )
    self.l1 = tf.layers.dense(inputs = self.x,units = 24, activation = tf.nn.tanh)
    self.l3 = tf.layers.dense(inputs = self.l1,units = 48,activation = tf.nn.tanh)
    self.l2 = tf.layers.dense(inputs = self.l1,units = nact, activation = None)
    self.loss = tf.nn.l2_loss(self.y-self.l2)
    self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss)
    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())
  def remember(self,state,action,reward,next_state,done):
    self.memory.append((state,action,reward,next_state,done))
  def choose_action(self,s,epsilon):
    if np.random.random() <= epsilon:
      return np.random.randint(0,self.nact)
    else :
      return np.argmax(self.sess.run(self.l2,feed_dict = {self.x:s[np.newaxis,:]})[0])
  def replay(self,batch_size):
    xbatch,ybatch = [],[]
    minibatch = random.sample(self.memory,min(len(self.memory),batch_size))
    for state,action,reward,nextstate,done in minibatch:
      state = state[np.newaxis,:]
      nextstate = nextstate[np.newaxis,:]
      ytarget = self.sess.run(self.l2,feed_dict={self.x:state})
      ytarget[0][action] = reward if done else reward + self.gamma*np.max(self.sess.run(self.l2,feed_dict={self.x:nextstate})[0])
      xbatch.append(state[0])
      ybatch.append(ytarget[0])
    xbatch = np.array(xbatch)
    ybatch = np.array(ybatch)
    self.sess.run(self.optimizer,feed_dict={self.x:xbatch,self.y:ybatch})
  def run(self):
    running_reward = None
    scores = deque(maxlen=100)
    for episode in range(1,1001):
      state = self.env.reset()
      done = False
      reward_sum = 0
      while not done:
        xin = state
        action = self.choose_action(xin,self.epsilon)
        #self.env.render()
        nextstate,reward,done,_ = self.env.step(action)
        self.remember(xin,action,reward,nextstate,done)
        state = nextstate
        reward_sum += reward
      if done:
        scores.append(reward_sum)
        self.replay(100)
        self.epsilon = max(self.epsilon_decay*self.epsilon,0.01)
        print episode,reward_sum
        print 'ep {}: reward: {}, mean reward: {:3f}'.format(episode, reward_sum, np.mean(scores))

solver = DQNSolver(gym.make('CartPole-v0'),4,2)
solver.run()
