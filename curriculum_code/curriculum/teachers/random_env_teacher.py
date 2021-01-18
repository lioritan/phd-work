from random import choice

from stable_baselines3.common.type_aliases import GymEnv

from curriculum.teacher import Teacher


class RandTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)


    def generate_task(self) -> GymEnv:
        r_param = choice(["easy", "hard"])

        pass

    def update_teacher_policy(self):
        pass
