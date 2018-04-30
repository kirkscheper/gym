import numpy as np
import gym
from gym import spaces
from datetime import datetime

from time import sleep
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
import Queue
import threading
import subprocess

class Simulation():
    
    def __init__(self):
        self.p = None
        
    def start(self, x, y, psi):
        # TODO send initial settings to simulation
        self.p = subprocess.Popen("../EvolutionaryLearning/EL", stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False, universal_newlines=True, close_fds=True)
        # set the O_NONBLOCK flag of p.stdout file descriptor:
        flags = fcntl(self.p.stdout, F_GETFL) # get current p.stdout flags
        fcntl(self.p.stdout, F_SETFL, flags | O_NONBLOCK)
        
    def end(self):
        if self.p is not None and self.p.poll() is None:
            self.p.stdout.flush()
            self.p.stdin.flush()
            self.p.kill()
            self.p.wait()

    def write_action(self, action):
        try:
            self.p.stdin.write(str(action) + "\n")
            self.p.stdin.flush()
        except IOError:
           return
        
    def read_state(self):
        result = None
        errors = 0
        while result is None:
            try:
                result = self.p.stdout.readline().strip()
            #except OSError:
                # the os throws an exception if there is no data
            #    print '[No more data]'
            except IOError:
                errors += 1
                # print 'not ready'
                if errors > 4:
                    self.end()
                    sleep(0.1)
                    self.start(2,2,0)
                    errors = 0
                    
                sleep(0.05)
        
        result = np.array([float(i) for i in result.split()])
        return result

class DelflyEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 25
    }
        
    def __init__(self):
        
        self.viewer = None
        
        self.max_angle = 2*0.523809524    # 30 deg
        self.action_low = np.array([-self.max_angle, 0.04])
        self.action_high = np.array([self.max_angle, 5])
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)    # angle offset to track
        
        #action_low = -self.max_angle
        #action_high = self.max_angle
        #self.action_space = spaces.Box(low=action_low, high=action_high, shape=(1,))    # angle offset to track
        
        _low = np.array([0,0,0,0,0,0,0,0,0,0,-3.142857143])    # apple detector location, apple detector size, average disparity
        _high = np.array([16,16,16,16,16,16,16,16,16,16,3.142857143])
        self.observation_space = spaces.Box(low = _low, high = _high)

        self.observation = np.array([0,0,0,0,0,0,0,0,0,0,0])
        self.state = None

        self._seed()
        self._reset()
        self.done = True
        
        self.sim = Simulation()
        
        self.poles = np.array([])
        
        #return observation, reward, done, info
        
    def _step(self, action):
        if self.done is True:
            self.sim.end()
            # reinitialise simulation with new initial conditions
            self.sim.start(2,2,0)
            self.done = False
            self.poles = self.sim.read_state()
            
        # run sim step
        orig_action = action
        action = np.clip(action, self.action_low, self.action_high)

        for a in action:
            self.sim.write_action(a)
        
        # read observations
        self.state = self.sim.read_state()
        
        if self.state.size == 0 or self.state[11] >= 0 or self.state[11] < -10:
            self.done = True

        if self.state.size >= 12:
            self.observation = self.state[0:11]
            self.reward = self.state[11]    # -distance to goal in dm
        #print self.reward, (abs(orig_action[0]) > self.max_angle), 0.1/(abs(orig_action[1]+0.1))
        self.reward = self.reward - (abs(orig_action[0]) > self.max_angle) - 0.1/(abs(orig_action[1]+0.1))   # penalize large actions
        #print self.reward
        
        return self.observation, self.reward, self.done, {}
        
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
        
        appleWidth = 0.5*scale
        
        delflyWidth = 0.3*scale
        delflyHeight = 0.3*scale
        
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
            
            lof = rendering.Line(start=(0.0, 0.0), end=(5*delflyWidth*0.5, 5*delflyWidth*0.866))   # line of sight
            lof.add_attr(self.delflyTrans)
            self.viewer.add_geom(lof)
            
            lof = rendering.Line(start=(0.0, 0.0), end=(-5*delflyWidth*0.5, 5*delflyWidth*0.866))   # line of sight
            lof.add_attr(self.delflyTrans)
            self.viewer.add_geom(lof)
            
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            set_point = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            set_point.set_color(.8,.6,.4)
            self.set_pointtrans = rendering.Transform(translation=(0, 0))
            set_point.add_attr(self.set_pointtrans)
            self.viewer.add_geom(set_point)
            
            self.poleTrans = []

        if self.state is None: return None
        
        if self.poles.size > 0:
            if len(self.poleTrans) is not self.poles.size / 2:
                self.poleTrans = []
                for i in range(0,self.poles.size/2):
                    pole = rendering.make_circle(appleWidth/2)
                    pole.set_color(.5,.5,.8)
                    self.poleTrans.append(rendering.Transform(translation=(0,0)))
                    pole.add_attr(self.poleTrans[i])
                    self.viewer.add_geom(pole)
            # add poles
            for i in range(0,self.poles.size/2):
                self.poleTrans[i].set_translation(self.poles[i*2]*scale, self.poles[i*2 + 1]*scale)
            self.poles = np.array([])
            return None

        if self.state.size < 14: return None

        x = self.state
        self.delflyTrans.set_translation(x[12]*scale, x[13]*scale)
        self.delflyTrans.set_rotation(x[10]-1.571428571)
        
        self.set_pointtrans.set_translation(x[12]*scale, x[13]*scale)
        self.set_pointtrans.set_rotation(x[14]-1.571428571)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
