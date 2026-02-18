"""ManipBench extensions — importing this module registers all custom components.

Subpackages (embodiments, tasks, scenes, policies, devices) use decorators
such as ``@register_asset`` to register themselves into Isaac Lab Arena's
global registries.  Simply importing this package triggers that registration.
"""

from . import embodiments  # noqa: F401
from . import tasks  # noqa: F401
from . import scenes  # noqa: F401
from . import policies  # noqa: F401
from . import devices  # noqa: F401
