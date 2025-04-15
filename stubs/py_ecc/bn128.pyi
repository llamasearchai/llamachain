"""
Type stubs for py_ecc.bn128 module.

This file helps type checkers like Pylance understand the interface of py_ecc.bn128.
"""

from typing import Any, Callable, List, Tuple, Union

# Type aliases
FQ = Any  # Field element
FQ2 = Any  # Quadratic extension field element
FQ12 = Any  # 12th-degree extension field element
G1Point = Tuple[FQ, FQ]  # Point on the elliptic curve E(FQ)
G2Point = Tuple[FQ2, FQ2]  # Point on the twisted curve E'(FQ2)

# Constants
curve_order: int

# G1 operations
def G1(x: Any, y: Any) -> G1Point: ...
def add(p1: G1Point, p2: G1Point) -> G1Point: ...
def multiply(pt: G1Point, n: int) -> G1Point: ...
def neg(pt: G1Point) -> G1Point: ...
def eq(p1: G1Point, p2: G1Point) -> bool: ...
def normalize(pt: G1Point) -> G1Point: ...
def is_on_curve(pt: G1Point) -> bool: ...

# G2 operations
def G2(x_c0: Any, x_c1: Any, y_c0: Any, y_c1: Any) -> G2Point: ...
def add2(p1: G2Point, p2: G2Point) -> G2Point: ...
def multiply2(pt: G2Point, n: int) -> G2Point: ...
def neg2(pt: G2Point) -> G2Point: ...
def eq2(p1: G2Point, p2: G2Point) -> bool: ...
def normalize2(pt: G2Point) -> G2Point: ...
def is_on_curve2(pt: G2Point) -> bool: ...

# Pairing operations
def pairing(p1: G1Point, p2: G2Point) -> FQ12: ...
def final_exponentiate(p: FQ12) -> FQ12: ...

# Curve operations
def curve_func(a: Any, b: Any) -> Callable[[FQ, FQ], FQ]: ...
def twisted_curve_func(a: Any, b: Any) -> Callable[[FQ2, FQ2], FQ2]: ...

# Field operations
def field_modulus() -> int: ...
def FQ2_one() -> FQ2: ...
def FQ12_one() -> FQ12: ...
