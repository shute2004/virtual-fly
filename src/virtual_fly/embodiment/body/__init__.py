"""Versioned FlyBody physical seams.

Version numbers are historical branches, not a linear inheritance chain.

    v3
    └─ v4
       ├─ v5
       ├─ v6
       └─ v7

    v8 = v6 + v7 neutral-trim mixin
"""

from .v3 import FlyBodyV3MuscleAdapter, FlyBodyV3NeuromuscularAdapter
from .v4 import FlyBodyV4MuscleAdapter, FlyBodyV4NeuromuscularAdapter
from .v5 import FlyBodyV5MuscleAdapter, FlyBodyV5NeuromuscularAdapter
from .v6 import FlyBodyV6MuscleAdapter, FlyBodyV6NeuromuscularAdapter
from .v7 import FlyBodyV7MuscleAdapter, FlyBodyV7NeuromuscularAdapter
from .v8 import FlyBodyV8MuscleAdapter, FlyBodyV8NeuromuscularAdapter

BODY_ADAPTERS = {
    "v3": FlyBodyV3NeuromuscularAdapter,
    "v4": FlyBodyV4NeuromuscularAdapter,
    "v5": FlyBodyV5NeuromuscularAdapter,
    "v6": FlyBodyV6NeuromuscularAdapter,
    "v7": FlyBodyV7NeuromuscularAdapter,
    "v8": FlyBodyV8NeuromuscularAdapter,
}

__all__ = [
    "BODY_ADAPTERS",
    "FlyBodyV3MuscleAdapter", "FlyBodyV3NeuromuscularAdapter",
    "FlyBodyV4MuscleAdapter", "FlyBodyV4NeuromuscularAdapter",
    "FlyBodyV5MuscleAdapter", "FlyBodyV5NeuromuscularAdapter",
    "FlyBodyV6MuscleAdapter", "FlyBodyV6NeuromuscularAdapter",
    "FlyBodyV7MuscleAdapter", "FlyBodyV7NeuromuscularAdapter",
    "FlyBodyV8MuscleAdapter", "FlyBodyV8NeuromuscularAdapter",
]
