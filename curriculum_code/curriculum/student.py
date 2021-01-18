import abc


class Student(abc.ABC):
    @abc.abstractmethod
    def get_action(self, obs, env, deterministic=False):
        pass

    @abc.abstractmethod
    def _record_action(self, obs, action, reward, next_obs, done):
        pass

    @abc.abstractmethod
    def _handle_done_signal(self):
        pass

    @abc.abstractmethod
    def _after_episode(self):
        pass

    @abc.abstractmethod
    def _before_episode(self):
        pass

    def train_episode(self, env, max_episode_length):
        # reset episode-specific variables
        obs = env.reset()
        done = False

        # collect experience by acting in the environment
        self._before_episode()
        i = 0
        while i < max_episode_length and not done:
            # get observation, act in the environment
            action = self.get_action(obs, env)  # TODO: handle recurrent policy (r_state, mask)
            old_obs = obs
            obs, reward, done, _ = env.step(action)  # TODO: add noise, scale, clip
            i += 1
            self._record_action(old_obs, action, reward, obs, done)

            if done:
                self._handle_done_signal()
        self._after_episode()
