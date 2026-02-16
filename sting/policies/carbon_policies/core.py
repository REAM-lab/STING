# -------------
# Import python packages
# --------------
from dataclasses import dataclass
import logging

# -------------
# Import sting code
# --------------
from sting.system.component import Component

# Set up logging
logger = logging.getLogger(__name__)

@dataclass(slots=True)
class CarbonPolicy(Component):
    """Class representing an carbon policy constraint. It applies for the full model horizon."""

    carbon_cap_tonneCO2peryear: float

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

