

class Teacher(object):
    def __init__(self, teacher_parameters, environment_parameters, seed=None):
        pass # TODO: environment_parameters -> generator? selector? how to API?

    def train_single_episode(self, student, max_episode_length):
        # TODO: generate task (using history)
        # TODO: give the student an episode? or k actions?
        # TODO: observe reward/trajectory, put in history
        # TODO: do parameter updates
        pass

    def test_single_episode(self, student, test_task, max_episode_length):
        # TODO: give the student an episode
        # TODO: observe reward
        pass #TODO