import gym
import procgen

# TODO: add optional shaped reward, discount -> in the curriculum
# TODO: customize the environments manually somehow?
# TODO: make a "get_train", "get_test"?


class ProcgenEnv(gym.Env):
    def render(self, mode='human'):
        self.render(mode)

    def __init__(self, env_config):
        env_name = env_config["name"]
        self.env_instance = gym.make(f"procgen:{env_name}",
                                     distribution_mode=env_config["distribution_mode"],  # default hard
                                     num_levels=env_config["num_levels"])  # default 0 (unlimited)
        self.action_space = self.env_instance.action_space
        self.observation_space = self.env_instance.observation_space

    def reset(self):
        return self.env_instance.reset()

    def step(self, action):
        return self.env_instance.step(action)
