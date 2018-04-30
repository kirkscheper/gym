import numpy as np
import gym
from gym import spaces
from datetime import datetime


import Queue
import threading
import subprocess

class Simulation():
    
    def __init__(self):
        self.p = None
        
    def start(self, x, y, psi):
        # TODO send initial settings to simulation
        
        self.p = subprocess.Popen("../EvolutionaryLearning/EL", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, universal_newlines=True)
        
    def write_action(self, action):
        try:
            self.p.stdin.write(str(action[0]) + "\n")
            self.p.stdin.flush()
        except IOError:
           return
        
    def read_state(self):
        result = self.p.stdout.readline().strip()
        self.p.stdout.flush()
        
        result = np.array([float(i) for i in result.split()])
            
        return result

class DelflyEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
        
    def __init__(self):
        
        self.max_angle = 1.047619048    # 60 deg
        #action_low = np.array([-self.max_angle, 0.1])
        #action_high = np.array([self.max_angle, 2.])
        #self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,))    # angle offset to track
        
        action_low = -self.max_angle
        action_high = self.max_angle
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(1,))    # angle offset to track
        
        _low = np.array([-self.max_angle, 0, 0])    # apple detector location, apple detector size, average disparity
        _high = np.array([self.max_angle, 128*128, 16])
        self.observation_space = spaces.Box(low = _low, high = _high)

        self.observation = np.array([0,0,0])
        self.state = None

        self._seed()
        self._reset()
        self.done = True
        
        self.sim = Simulation()
        
        self.viewer = None
        
        #return observation, reward, done, info
        
    def _step(self, action):
        if self.done is True:
            if self.sim.p is not None:
                self.sim.p.kill()
            # reinitialise simulation with new initial conditions
            self.sim.start(2,2,0)
            self.done = False
        
        # run sim step
        orig_action = action
        action = np.clip(action, -self.max_angle, self.max_angle)

        self.sim.write_action(action)
        
        # read observations
        self.state = self.sim.read_state()
        
        if self.state.size == 0:
            self.done = True
            reward = 0
        else:
            self.observation = self.state[0:3]
            reward = self.state[3]
            
        reward = reward - 0.001*abs(orig_action[0])   # penalize large actions
        
        return self.observation, reward, self.done, {}
        
    def _reset(self):
        return self.observation
    
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = 8
        scale = screen_width/world_width
        appleWidth = 10.0
        
        delflyWidth = 30.0
        delflyHeight = 30.0
        
        polewidth = 5.0
        polelen = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            # add Delfly
            l,r,t,b = -delflyWidth/2, delflyWidth/2, delflyHeight/2, -delflyHeight/2
            delfly = rendering.FilledPolygon([(l,b), (0,t), (0,t), (r,b)])  # triangle
            self.delflyTrans = rendering.Transform()
            delfly.add_attr(self.delflyTrans)
            self.viewer.add_geom(delfly)
            
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            
            # add apple
            apple = rendering.make_circle(appleWidth/2)
            apple.set_color(.5,.5,.8)
            self.appleTrans = rendering.Transform()
            apple.add_attr(self.appleTrans)
            self.viewer.add_geom(apple)

        if self.state is None: return None
        if self.state.size < 10: return None

        x = self.state
        self.delflyTrans.set_translation(x[4]*scale, x[5]*scale)
        self.delflyTrans.set_rotation(x[6]-1.571428571)
        
        self.appleTrans.set_translation(x[7]*scale, x[8]*scale)
        
        self.poletrans.set_translation(x[4]*scale, x[5]*scale)
        self.poletrans.set_rotation(x[9]-1.571428571)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
