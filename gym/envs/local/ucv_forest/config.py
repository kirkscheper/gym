import inspect, os

class Config:

    # Training Settings
    LOAD_MODEL = True
    MODEL_PATH = './model'
    MODEL_SAVE_FREQ = 50
    VERBOSITY = 1

    # Basic RL settings
    MAX_EPISODE_LENGTH = 750
    ACTIONS = 2
    NUM_WORKERS = 1
    MAX_EPISODES = 16000
    GOAL_ON = True
    FOV = 90

    # Auxiliary tasks
    AUX_TASK_D2 = True

    # Simulator settings
    HOST = 'localhost'
    PORT = 9001
    SIM_DIR = os.path.split(os.path.realpath(__file__))[0] + '/ucv-pkg-outdoor-xenial/LinuxNoEditor/outdoor_lite/Binaries/Linux/'
    SIM_NAME = 'outdoor_lite'

    RANDOM_SPAWN_LOCATIONS = True
    MAP_X_MIN = -4000
    MAP_X_MAX = 4000
    MAP_Y_MIN = -4000
    MAP_Y_MAX = 4000

