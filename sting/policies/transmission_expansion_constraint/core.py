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
class TransmissionExpansionConstraint(Component):
    """Class representing an transmission capacity constraint over full system. It applies for the full model horizon."""

    built_transmission_capacity_cap_MW: float

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
