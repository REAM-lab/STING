# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass, field

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.modules.small_signal_modeling import SmallSignalModel


@dataclass(slots=True)
class ReducedOrderModel:
    ssm: SmallSignalModel

    def