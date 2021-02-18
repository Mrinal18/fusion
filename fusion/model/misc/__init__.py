from collections import namedtuple


ModelOutput = namedtuple(
    'ModelOutput',
    ['latents', 'attrs']
)

__all__ = [
    'ModelOutput'
]
