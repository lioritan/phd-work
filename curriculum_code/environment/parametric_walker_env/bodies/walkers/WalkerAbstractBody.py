from environment.parametric_walker_env.bodies.AbstractBody import AbstractBody
from environment.parametric_walker_env.bodies.BodyTypesEnum import BodyTypesEnum


class WalkerAbstractBody(AbstractBody):
    def __init__(self, scale, motors_torque, nb_steps_under_water):
        super(WalkerAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.WALKER
        self.nb_steps_can_survive_under_water = nb_steps_under_water