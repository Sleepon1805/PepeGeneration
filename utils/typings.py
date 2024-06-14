import abc
import torch
from typing import Tuple
from jaxtyping import Float, Float32, jaxtyped
from beartype import beartype

"""
Docs for jaxtyping:
https://docs.kidger.site/jaxtyping/
"""

RUNTIME_TYPECHECKS = True


# torch.Tensor of dtype float32 and shape (num_images, num_channels, img_size, img_size)
TrainImagesType = Float32[torch.Tensor, "num_images num_channels img_size img_size"]

# torch.Tensor of dtype float32 and shape (num_images,) or (1,)
BatchedFloatType = Float[torch.Tensor, "#num_images"] | float

# torch.Tensor of dtype float32 and shape (num_images, condition_size) or None
ConditionType = Float32[torch.Tensor, "num_images condition_size"] | None

# One batch of data: images and labels
BatchType = Tuple[TrainImagesType, ConditionType]


def typecheck(func):
    """
    Decorator that applies jax type checking to a function.
    :param func:
    :return:
    """
    typechecker = beartype if RUNTIME_TYPECHECKS else None
    result = jaxtyped(func, typechecker=typechecker)
    return result


class _TypeCheckedMetaClass(type):
    def __new__(cls, name, bases, local):
        for attr in local:
            value = local[attr]
            if callable(value):
                local[attr] = typecheck(value)
        return type.__new__(cls, name, bases, local)


class TypeChecked(metaclass=_TypeCheckedMetaClass):
    """
    Class that applies jax type checking to all its methods.
    """
    pass


class _ABCTypeCheckedMetaClass(abc.ABCMeta):
    def __new__(cls, name, bases, local):
        super().__new__(cls, name, bases, local)
        for attr in local:
            value = local[attr]
            if callable(value):
                local[attr] = typecheck(value)
        return type.__new__(cls, name, bases, local)


class ABCTypeChecked(abc.ABC, metaclass=_ABCTypeCheckedMetaClass):
    """
    Abstract class that applies jax type checking to all its methods.
    """
    pass


if __name__ == "__main__":
    import abc

    class TestClassA(ABCTypeChecked):
        def __init__(self):
            super().__init__()

        @abc.abstractmethod
        def test_func(self, x: TrainImagesType, y: TrainImagesType) -> TrainImagesType:
            pass


    class TestClassB(TestClassA):
        def __init__(self):
            super().__init__()

        def test_func(self, x: TrainImagesType, y: TrainImagesType) -> TrainImagesType:
            return x - y


    x = torch.randn((32, 1, 64, 64))
    y = torch.randn((32, 6, 64, 64))
    res = TestClassB().test_func(x, y)
    print(res.shape)
