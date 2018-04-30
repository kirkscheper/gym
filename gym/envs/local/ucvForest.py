from __future__ import print_function
import math
import numpy as np
try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO
import re
from PIL import Image
from random import randint, sample
import time
import subprocess
import os
from scipy.ndimage import zoom
from skimage.transform import resize
import yaml
from unrealcv import Client

from .ucv_forest.config import Config

import matplotlib as mpl
import os
if "DISPLAY" not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input

class ucvForest(gym.GoalEnv):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 25
    }

    """ Class for interacting with an UE4 Game packaged with the UnrealCV plugin
        To make this sim work as a gym.Env use: env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'desired_goal'])
    """

    def __init__(self, mode=None, visualize=0, preprocess_input=False):
        # initialise settings
        self.visualize = visualize

        if self.visualize > 1:
            self.input_fig = plt.figure()
            self.input_ax = self.input_fig.add_subplot(111)

        if self.visualize > 2:
            self.conv_fig = plt.figure()
            self.conv_axes = None

        self.speed = 50.0  # cm/step
        self.angle = 5.0  # degrees/step

        # initialise env variables
        self.trajectory = []
        self.port = Config.PORT
        Config.PORT += 1
        self.name = 'worker_' + str(self.port)

        # navigation goal direction
        self.goal = None
        self.goal_vector = None

        # list of start and goal locations
        self.locations = []
        if Config.RANDOM_SPAWN_LOCATIONS:
            self.locations = None
        else:
            with open(Config.SIM_DIR + 'locations.yaml', 'r') as loc_file:
                self.locations = yaml.load(loc_file)

        # RL rewards
        self.goal_direction_reward = self.speed / 100.
        self.crash_reward = -self.speed / 10.

        # gym settings
        self.viewer = None

        self.max_angle = 1. # ~60 deg
        self.max_skip = 5. 

        action_low = np.array([-self.max_angle, -1.])
        action_high = np.array([self.max_angle, 1.])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype='float32')    # angle offset to track

        self.reward_type = 'sparse'
        self.distance_threshold = self.max_skip * self.speed

        # load UE4 game
        self.sim = None
        self.client = None
        self.should_stop = False
        self.mode = mode

        self.preprocess_input = preprocess_input
        if self.preprocess_input:
            self.preprocess_model = VGG16(weights='imagenet', include_top=False)
        else:
            self.preprocess_model = None

        self.seed()
        obs = self.reset()

        if self.preprocess_input:
            self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(Config.MAP_X_MIN, Config.MAP_X_MAX, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(Config.MAP_X_MIN, Config.MAP_X_MAX, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-math.pi, math.pi, shape=obs['observation'].shape, dtype='float32'),
            ))
        else:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(Config.MAP_X_MIN, Config.MAP_X_MAX, shape=obs['desired_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(Config.MAP_X_MIN, Config.MAP_X_MAX, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Tuple((spaces.Box(-math.pi, math.pi, shape=obs['observation'][0].shape, dtype='float32'),
                                          spaces.Box(low=-1., high=1., shape=obs['observation'][1].shape, dtype='float32'))),
                ))

    @staticmethod
    def _read_npy(res):
        return np.load(BytesIO(res))

    @staticmethod
    def _read_png(res):
        img = Image.open(BytesIO(res))
        return np.asarray(img)

    @staticmethod
    def quantize_depth(depth_image):
        """ Depth classes """

        bins = [0, 1, 2, 3, 4, 5, 6, 7]  # TODO: better depth bins
        out = np.digitize(depth_image, bins) - np.ones(depth_image.shape, dtype=np.int8)
        return out

    def random_start_location(self, heading):
        collision_at_start = True
        while collision_at_start:
            # spawn location is ok, if we can move forward a bit without colliding
            start_x = randint(Config.MAP_X_MIN, Config.MAP_X_MAX)
            start_y = randint(Config.MAP_Y_MIN, Config.MAP_Y_MAX)
            self.request('vset /camera/0/pose {} {} {} {} {} {}'.format(start_x, start_y, 150., 0., heading, 0.))
            step = 100.  # cm
            small_step_forward = (start_x + step * math.cos(math.radians(heading)),
                                  start_y + step * math.sin(math.radians(heading)), 150.)
            self.request('vset /camera/0/moveto {} {} {}'.format(*small_step_forward))
            final_loc = [round(float(v), 2) for v in self.request('vget /camera/0/location').split(' ')]
            if final_loc == [round(v, 2) for v in small_step_forward]:
                # acceptable start location found
                collision_at_start = False
            else:
                time.sleep(0.1)

            self.request('vset /camera/0/pose {} {} {} {} {} {}'.format(start_x, start_y, 150., 0., heading, 0.))

        return start_x, start_y

    def goal_distance(self, location, goal):
        assert location.shape == goal.shape
        return np.linalg.norm(location - goal, axis=-1)

    def goal_direction(self):
        """ Producing goal direction input for the agent (rad). """

        location = np.array(self.trajectory[-1]['location'])
        goal_vector = np.subtract(self.goal, location)
        goal = math.atan2(goal_vector[1], goal_vector[0])

        hdg = math.radians(self.trajectory[-1]['rotation'][1])
        relative = goal - hdg

        # wrap to [-180, 180]
        while relative < -math.pi:
            relative += 2*math.pi
        while relative > math.pi:
            relative -= 2*math.pi

        return np.expand_dims(relative, 0)

    def save_trajectory(self):
        """ Save trajectory for evaluation. """

        filename = './trajectory_{}.yaml'.format(self.name)
        with open(filename, 'a+') as trajectory_file:
            traj_dict = {'traj': self.trajectory,
                         'goal': [float(self.goal[0]), float(self.goal[1])],
                         }  # TODO: add rewards
            yaml.dump([traj_dict], stream=trajectory_file, default_flow_style=False)


    # Sim methods
    # ----------------------------
    def set_port(port, sim_dir):
        with open(sim_dir + 'unrealcv.ini', 'w') as ini_file:
            print('[UnrealCV.Core]', file=ini_file)
            print('Port={}'.format(str(port)), file=ini_file)
            print('Width=224', file=ini_file)
            print('Height=224', file=ini_file)
            print('FOV=90.0', file=ini_file)
            print('EnableInput=False', file=ini_file)
            print('EnableRightEye=False', file=ini_file)

    def start_sim(self, restart=False):
        """ Starting game and connecting the UnrealCV client """

        if restart:
            # disconnect and terminate if restarting
            self.shut_down()

        if self.sim is None:
            self.set_port(self.port, Config.SIM_DIR)
            print('[{}] Connection attempt on PORT {}.'.format(self.name, self.port))
            with open(os.devnull, 'w') as fp:   # Sim messages on stdout are discarded
                exe = Config.SIM_DIR + Config.SIM_NAME
                #print(exe)
                #if "CUDA_VISIBLE_DEVICES" in os.environ:
                #    exe = "CUDA_VISIBLE_DEVICES={} ".format(os.environ["CUDA_VISIBLE_DEVICES"]) + exe
                #    print exe
                self.sim = subprocess.Popen(exe, stdout=fp)
            time.sleep(3)
            self.client = Client((Config.HOST, self.port))
            

        if not self.client.isconnected():
            time.sleep(2)
            self.client.connect()

        if not self.client.isconnected():
            return False

        return True

    def shut_down(self):

        """ Disconnect client and terminate the game. """

        if self.client.isconnected():
            self.client.disconnect()
        if self.sim is not None:
            while self.sim.poll() is None:
                self.sim.terminate()
                time.sleep(0.2)
            self.sim = None

    def get_pos(self, print_pos=False):
        """ Get the last position from the stored trajectory, if trajectory is empty then request it from the sim. """

        if len(self.trajectory) == 0:
            rot = np.array([float(v) for v in self.request('vget /camera/0/rotation').split(' ')])
            loc = np.array([float(v) for v in self.request('vget /camera/0/location').split(' ')])
            self.trajectory.append(dict(location=loc, rotation=rot))
        else:
            loc = self.trajectory[-1]["location"]
            rot = self.trajectory[-1]["rotation"]

        if print_pos:
            print('Position x={} y={} z={}'.format(*loc))
            print('Rotation pitch={} heading={} roll={}'.format(*rot))

        return loc, rot

    def reset_agent(self):
        """ Reset the agent to continue interaction in the state where it was interrupted. """

        new_loc = self.trajectory[-1]["location"]
        new_rot = self.trajectory[-1]["rotation"]
        res1 = self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
        assert res1
        res2 = self.request('vset /camera/0/location {:.2f} {:.2f} {:.2f}'.format(*new_loc))
        assert res2

        return

    def request(self, message):
        """ Send request with UnrealCV client, if unsuccessful restart sim. """

        res = self.client.request(message)
        # if res in 'None', try restarting sim
        while not res:
            print('[{}] sim error while trying to request {}'.format(self.name, message))
            success = self.start_sim(restart=True)
            if success:
                res = self.client.request(message)

        return res

    def preprocess_observation(self, obs):
        assert obs.ndim is 3
        if self.preprocess_model is None:
            return obs

        #resize image for vgg16
        if obs.shape[:2] is not (224,224):
            obs = resize(obs, (224,224))

        features = self.preprocess_model.predict(np.expand_dims(obs,0))[0]

        #print(np.min(features), np.max(features))
        if self.visualize > 2:
            if self.conv_axes is None:
                self.conv_axes = []
                n = features.shape[2]
                n = int(np.ceil(np.sqrt(n)))
                for i in range(features.shape[2]):
                    self.conv_axes.append(self.conv_fig.add_subplot(n,n,i+1))

            max_f = np.max(features)
            if max_f > 1e-5:
                plot_f = features / max_f
            else:
                plot_f = features
            for i in range(plot_f.shape[2]):
                self.conv_axes[i].cla()
                self.conv_axes[i].imshow(plot_f[:,:,i], cmap='gray')
            plt.pause(0.005)

        return features.flatten()

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._step_callback()
        obs = self._get_obs()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'is_collision': self._is_collision(),
        }
        done = False #info['is_success'] or info['is_collision']
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()

        return obs

    #def close(self):
    #    if self.viewer is not None:
    #        self.viewer.finish()
    #       self.viewer = None

    def close(self):
        if self.visualize > 1:
            plt.close(self.input_fig)
        if self.visualize > 2:
            plt.close(self.conv_fig)

        self.shut_down()
        self.render(close=True)

    def _get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            screen_width = 800
            screen_height = 800
            world_width = 100
            self.render_scale = screen_width/world_width

            self.viewer = rendering.Viewer(screen_width, screen_height)

            delflyWidth = 0.3*self.render_scale
            delflyHeight = 0.3*self.render_scale

            sp_indicator_w = 5.0
            sp_indicator_l = 30.0

            # add Delfly
            l,r,t,b = -delflyWidth/2, delflyWidth/2, delflyHeight/2, -delflyHeight/2
            delfly = rendering.FilledPolygon([(l,b), (0,t), (0,t), (r,b)])  # triangle
            self.delflyTrans = rendering.Transform()
            delfly.add_attr(self.delflyTrans)
            self.viewer.add_geom(delfly)

            lof = rendering.Line(start=(0.0, 0.0), end=(5*delflyWidth*math.sin(Config.FOV/2), 5*delflyWidth*math.cos(Config.FOV/2)))   # line of sight
            lof.add_attr(self.delflyTrans)
            self.viewer.add_geom(lof)

            lof = rendering.Line(start=(0.0, 0.0), end=(-5*delflyWidth*math.sin(Config.FOV/2), 5*delflyWidth*math.cos(Config.FOV/2)))   # line of sight
            lof.add_attr(self.delflyTrans)
            self.viewer.add_geom(lof)

            l,r,t,b = -sp_indicator_w/2,sp_indicator_w/2,sp_indicator_l-sp_indicator_w/2,-sp_indicator_w/2
            sp = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            sp.set_color(.8,.6,.4)
            self.sp_trans = rendering.Transform(translation=(0, 0))
            sp.add_attr(self.sp_trans)
            self.viewer.add_geom(sp)

            l,r,t,b = -10, 10, 10, -10
            goal = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])  # triangle
            self.goalTrans = rendering.Transform()
            goal.add_attr(self.goalTrans)
            self.viewer.add_geom(goal)

            self.line = []
            line = rendering.PolyLine(self.line, False)
            self.viewer.add_geom(line)

            self._viewer_setup()
        return self.viewer

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self._get_viewer()

        if self.trajectory is None: return None

        hdg = -math.radians(self.trajectory[-1]['rotation'][1])
        loc = (np.array(self.trajectory[-1]['location']) + 5000)*self.render_scale / 100

        self.delflyTrans.set_translation(loc[1], loc[0])
        self.delflyTrans.set_rotation(hdg)

        self.sp_trans.set_translation(loc[1], loc[0])
        self.sp_trans.set_rotation(hdg)

        if len(self.trajectory) <= 2:
            del self.line[:]
        self.line.append([loc[1], loc[0]])
        #self.viewer.draw_polyline((np.array([[t['location'][1], t['location'][0]] for t in self.trajectory]) + 5000)*self.render_scale / 100)
        self.goalTrans.set_translation((self.goal[1]+5000)*self.render_scale / 100, (self.goal[0]+5000)*self.render_scale / 100)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    # GoalEnv methods
# ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        collision_score = self.crash_reward * np.array(info['is_collision']).reshape(d.shape)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32) + collision_score
        else:
            return -d  + collision_score

        reward = 0
        step_size = np.linalg.norm(disp)
        norm_displacement = np.array(disp) / step_size
        norm_goal_vector = np.subtract(goal_location, prev_loc)\
                           / np.linalg.norm(np.subtract(goal_location, prev_loc))
        reward += (step_size / self.speed) * np.dot(norm_goal_vector, norm_displacement) * goal_direction_reward   # scale reward with step size
        reward -= 0.1*np.sum(np.array([action[i] < self.action_low[i] for i in range(0,len(self.action_low))]) * np.abs(self.action_low - action) + \
                         np.array([action[i] > self.action_high[i] for i in range(0,len(self.action_high))]) * np.abs(self.action_high - action))  # penalty for out of bounds action

        goal_distance

        return reward 

    # Extension methods
# ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        while not self.start_sim():
            pass

        # choose random respawn and goal locations, either randomly or from a list of predetermined locations
        random_heading = (0.0, randint(0, 360), 0.0) #(0.0, randint(0, 360), 0.0)
        if Config.RANDOM_SPAWN_LOCATIONS:
            start_x, start_y = self.random_start_location(random_heading[1])
            start_loc = (start_x, start_y, 150.)
        else:
            if start is None or goal is None:
                idx_start, idx_goal = sample(range(0, len(self.locations) - 1), 2)
            else:
                idx_start = start
            start_loc = (self.locations[idx_start]['x'], self.locations[idx_start]['y'], self.locations[idx_start]['z'])

        # reset trajectory
        self.trajectory = []
        loc = [float(v) for v in start_loc]
        rot = [float(v) for v in random_heading]
        self.trajectory.append(dict(location=np.array(loc), rotation=np.array(rot)))

        return True

    def _get_obs(self):
        """Returns the observation.
        """
        grayscale=False
        viewmode='lit'

        if viewmode == 'depth':
            res = self.request('vget /camera/0/depth npy')
            depth_image = self._read_npy(res)
            # resize 84x84 to 16x16, crop center 8x16
            depth_image = depth_image[21:63]
            depth_image = zoom(depth_image, [0.095, 0.19], order=1)
            image = self.quantize_depth(depth_image)
        else:
            res = self.request('vget /camera/0/lit png')
            rgba = self._read_png(res).copy().astype(np.float32)
            rgb = rgba[:, :, :3]
            rgb = preprocess_input(rgb, data_format='channels_last', mode='tf')
            if grayscale is True:
                image = np.mean(rgb, 2)
            else:
                image = rgb

        if self.visualize > 1:
            self.input_ax.cla()
            self.input_ax.imshow(image/2.+.5)     # scale to visible range
            plt.pause(0.005)
            #plt.savefig('image.png')
            #print('plot saved')

        obs = self.preprocess_observation(image)

        goal_dir = self.goal_direction()
        if obs.ndim == goal_dir.ndim:
            obs = np.concatenate((obs, goal_dir), axis=0)
        else:
            obs = [obs, goal_dir]

        return {
                'observation': obs.copy(),
                'achieved_goal': self.trajectory[-1]['location'].copy(),
                'desired_goal': self.goal.copy(),
                }

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        relative=True
        trans_cmd = np.zeros(3)
        rot_cmd = np.zeros(3)

        orig_action = action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if isinstance(action, str): # discrete action
            if action == 'left':
                trans_cmd[0] = self.speed
                rot_cmd[1] = -self.angle
            elif action == 'right':
                trans_cmd[0] = self.speed
                rot_cmd[1] = self.angle
            elif action == 'forward':
                trans_cmd[0] = self.speed
            elif action == 'backward':
                trans_cmd[0] = -self.speed
            else:
                print('Unknown action: ' + action)
        else: # continuous action
            trans_cmd[0] = self.speed * (action[1]/2. + 0.5) * self.max_skip    # scale action to output skip
            rot_cmd[1] = math.degrees(action[0])

        if relative:
            loc, rot = self.get_pos()
            new_rot = np.array([sum(x) % 360 for x in zip(rot, rot_cmd)])
            translation = [trans_cmd[0] * math.cos(math.radians(new_rot[1])), trans_cmd[0] * math.sin(math.radians(new_rot[1])), 0.0]
            new_loc = np.array([sum(x) for x in zip(loc, translation)])
        else:
            new_rot = rot_cmd
            new_loc = trans_cmd

        if (rot_cmd != np.zeros(3)).any() or not relative:
            res = self.request('vset /camera/0/rotation {:.3f} {:.3f} {:.3f}'.format(*new_rot))
            if res != 'ok':
                print('Requested rotation unsuccessful ', res)
            # assert(res == 'ok')
        if (trans_cmd != np.zeros(3)).any() or not relative:
            res = self.request('vset /camera/0/moveto {:.2f} {:.2f} {:.2f}'.format(*new_loc))
            if res != 'ok':
                print('Requested translation unsuccessful ', res)
            # assert (res == 'ok')

        self.trajectory.append(dict(location=new_loc, rotation=new_rot))

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        def check_circle_intersection(line_start, line_end, circle_loc, radius):
            # based on http://mathworld.wolfram.com/Circle-LineIntersection.html
            # convert everything to meteres to reduce magnitude of values
            line_start = np.array(line_start) / 100.
            line_end = np.array(line_end) / 100.
            circle_loc = np.array(circle_loc) / 100.
            radius /= 100.

            dx = line_end[0] - line_start[0]
            dy = line_end[1] - line_start[1]

            line_start -= circle_loc
            line_end   -= circle_loc
            dr2 = dx**2 + dy**2
            D = line_start[0]*line_end[1] - line_end[0]*line_start[1]

            if radius**2 * dr2 < D**2:
                return False

            y = -D*dx + np.abs(dy)*np.sqrt(radius**2*dr2 - D**2)/dr2

            return np.abs(y - line_start[1] - dy) < np.abs(dy)

        if self.goal_distance(achieved_goal, desired_goal) < self.distance_threshold:
            return True

        return False

    def _is_collision(self):
        """Indicates whether or not the agent collided with an obstacle.
        """
        collision = False
        cam_loc = [float(v) for v in self.request('vget /camera/0/location').split(' ')]
        if cam_loc != [round(v, 2) for v in self.trajectory[-1]['location']]:
            collision = True
            self.trajectory[-1]['location'] = cam_loc[:]    # TODO bounce back

        return collision

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        if Config.RANDOM_SPAWN_LOCATIONS:
            goal_x = randint(Config.MAP_X_MIN, Config.MAP_X_MAX)
            goal_y = randint(Config.MAP_Y_MIN, Config.MAP_Y_MAX)
            goal_location = np.array([goal_x, goal_y, 150.])
        else:
            # sample from list of possible start/end locations
            idx_goal = sample(xrange(len(self.locations)))
            goal_location = np.array([self.locations[idx_goal]['x'], self.locations[idx_goal]['y'], self.locations[idx_goal]['z']])

        return goal_location.copy()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
