from enum import Enum

from environment.parametric_walker_env.bodies.BodyTypesEnum import BodyTypesEnum
from environment.parametric_walker_env.bodies.walkers.BigQuadruBody import BigQuadruBody
from environment.parametric_walker_env.bodies.walkers.ClassicBipedalBody import ClassicBipedalBody
from environment.parametric_walker_env.bodies.walkers.HumanBody import HumanBody
from environment.parametric_walker_env.bodies.walkers.MillipedeBody import MillipedeBody
from environment.parametric_walker_env.bodies.walkers.ProfileChimpanzee import ProfileChimpanzee
from environment.parametric_walker_env.bodies.walkers.SmallBipedalBody import SmallBipedalBody
from environment.parametric_walker_env.bodies.walkers.SpiderBody import SpiderBody
from environment.parametric_walker_env.bodies.walkers.WheelBody import WheelBody


class BodiesEnum(Enum):
    small_bipedal = SmallBipedalBody
    classic_bipedal = ClassicBipedalBody
    big_quadru = BigQuadruBody
    spider = SpiderBody
    millipede = MillipedeBody
    wheel = WheelBody
    human = HumanBody
    profile_chimpanzee = ProfileChimpanzee

    @classmethod
    def get_body_type(self, body_name):
        if body_name in ['climbing_chest_profile_chimpanzee', 'climbing_profile_chimpanzee']:
            return BodyTypesEnum.CLIMBER
        elif body_name == 'fish':
            return BodyTypesEnum.SWIMMER
        elif body_name == 'amphibious_bipedal':
            return BodyTypesEnum.AMPHIBIAN
        else:
            return BodyTypesEnum.WALKER