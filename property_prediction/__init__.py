"""Property prediction package.

The alias keeps previously serialized reducers/models loadable after the
package was renamed from ``module_predict`` to ``property_prediction``.
New source code must import ``property_prediction`` directly.
"""

from __future__ import annotations

import sys


sys.modules.setdefault("module_predict", sys.modules[__name__])
