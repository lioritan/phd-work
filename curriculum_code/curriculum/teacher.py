from abc import ABC, abstractmethod


from curriculum.history import History
from curriculum.student import Student


class Teacher(ABC):
    def __init__(self, teacher_parameters, environment_parameters, history_parameters=None, seed=None):
        # is the parameter range/value set required here?
        # TODO: environment_parameters -> generator? selector? how to API?
        self.history = History(history_parameters)
        self.seed = seed

    def train_single_episode(self, student: Student, max_episode_length: int):
        training_task = self.generate_task()
        trajectory, rewards = self.train_student_on_task(student, training_task, max_episode_length)
        self.history.update(training_task, trajectory, rewards)
        self.update_teacher_policy()

    def train_student_on_task(self, student, training_task, max_episode_length):
        action_limit = max_episode_length
        student.train_episode(training_task, action_limit)

        # TODO: run student for action_limit steps. Can this be done with student.learn?
        #print(student)
        #print(student.replay_buffer)
        # TODO: return the trajectory and rewards
        return [], []

    @abstractmethod
    def generate_task(self):
        pass

    @abstractmethod
    def update_teacher_policy(self):
        pass
