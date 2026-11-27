"""User-facing analytical Boolean expressions.

This module is deliberately independent of Trimesh, OpenMC and TRIPOLI-5.
It describes *what part of space* is occupied by a solid, not how that solid
is rendered. The backend-independent CSG compiler later translates an
expression into analytical surfaces and half-spaces.

Public examples::

    fuel = Cylinder(0.61, 50.0)
    clad_outer = Cylinder(0.65, 50.0)
    cladding = clad_outer - fuel

    ring_water = (Cylinder(2.0, 50.0) - Cylinder(1.0, 50.0)) - pin_union

The operators construct a lazy expression tree. They do not execute mesh
booleans and they do not approximate curved surfaces with triangles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import Primitive, _transform, _rotation_xyz


@dataclass(frozen=True)
class _ShapeLeaf:
    """A primitive plus its rigid transform in the expression's local frame.

    A leaf stores a 4x4 matrix as a tuple rather than a NumPy array so the node
    is immutable and can safely be shared by several Boolean expressions.
    Transformations are accumulated here and applied only when CSG is compiled.
    """
    shape: Primitive
    matrix: tuple[float, ...]

    @classmethod
    def create(cls, shape: Primitive) -> "_ShapeLeaf":
        return cls(shape, tuple(float(v) for v in np.eye(4).ravel()))

    def matrix_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float).reshape(4, 4)


@dataclass(frozen=True)
class _BooleanNode:
    """One Boolean operation whose terms are other expression-tree nodes."""
    operation: str
    terms: tuple[object, ...]


ExpressionNode = _ShapeLeaf | _BooleanNode


def _coerce_node(value) -> ExpressionNode:
    """Convert a primitive or existing expression into an immutable AST node."""
    if isinstance(value, RegionExpression):
        return value._node
    if isinstance(value, Primitive):
        return _ShapeLeaf.create(value)
    raise TypeError(
        "A Boolean region operand must be an analytical Primitive or RegionExpression; "
        f"got {type(value).__name__}"
    )


def _transform_node(node: ExpressionNode, transform: np.ndarray) -> ExpressionNode:
    """Return a copy of ``node`` with ``transform`` applied to every leaf.

    Applying the transform to all leaves, rather than to a final triangulated
    object, is what keeps the expression analytical. For a primitive leaf the
    new matrix is ``transform @ existing_matrix``: the primitive's local point
    is first transformed by its existing placement, then by the new operation.
    For a Boolean node, the same transformation is recursively distributed to
    all operands; this is valid because rigid transformations commute with set
    union, intersection and difference.
    """
    if isinstance(node, _ShapeLeaf):
        composed = transform @ node.matrix_array()
        return _ShapeLeaf(node.shape, tuple(float(v) for v in composed.ravel()))
    return _BooleanNode(
        node.operation,
        tuple(_transform_node(term, transform) for term in node.terms),
    )


class RegionExpression:
    """Lazy Boolean expression over analytical MiniGeom primitives.

    ``&`` means intersection, ``|`` means union, ``-`` means set difference,
    and ``~`` means complement. Operator precedence follows Python's normal
    rules, so parentheses are recommended in expressions mixing several
    operators.

    Expressions are local-coordinate objects. When attached to a VolumeDef via
    ``VolumeDef.add_cell()``, they are transformed by that volume's world matrix
    and clipped to the volume's bounded domain by ``CSGCompiler``.
    """

    def __init__(self, node: ExpressionNode):
        self._node = node

    @classmethod
    def from_primitive(cls, shape: Primitive) -> "RegionExpression":
        """Create an expression representing the interior of one primitive."""
        return cls(_coerce_node(shape))

    def _binary(self, operation: str, other) -> "RegionExpression":
        right = _coerce_node(other)
        return RegionExpression(_BooleanNode(operation, (self._node, right)))

    def __and__(self, other) -> "RegionExpression":
        return self._binary("intersection", other)

    def __rand__(self, other) -> "RegionExpression":
        return as_region_expression(other)._binary("intersection", self)

    def __or__(self, other) -> "RegionExpression":
        return self._binary("union", other)

    def __ror__(self, other) -> "RegionExpression":
        return as_region_expression(other)._binary("union", self)

    def __sub__(self, other) -> "RegionExpression":
        return self._binary("difference", other)

    def __rsub__(self, other) -> "RegionExpression":
        return as_region_expression(other)._binary("difference", self)

    def __invert__(self) -> "RegionExpression":
        return RegionExpression(_BooleanNode("complement", (self._node,)))

    def translated(self, offset) -> "RegionExpression":
        """Return a copy translated in the expression's coordinate frame."""
        transform = _transform(offset, (0.0, 0.0, 0.0))
        return RegionExpression(_transform_node(self._node, transform))

    def rotated(self, angles, *, order: str = "xyz") -> "RegionExpression":
        """Return a copy rotated about the expression origin.

        The public primitive convention is XYZ Euler angles in degrees, composed
        as ``Rz @ Ry @ Rx``. Other Euler orders are intentionally rejected here
        to keep the transformation convention unambiguous across backends.
        """
        if order.lower() != "xyz":
            raise ValueError("Only XYZ Euler rotation order is supported")
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = _rotation_xyz(tuple(float(v) for v in angles))
        return RegionExpression(_transform_node(self._node, matrix))

    def transformed(self, matrix) -> "RegionExpression":
        """Return a copy transformed by a rigid 4x4 homogeneous matrix.

        Scaling and shear are rejected: OpenMC/AGORA surface transforms in this
        project are rigid transformations, and silently converting scaled
        quadrics would be error-prone. Rotation matrices must be orthonormal.
        """
        value = np.asarray(matrix, dtype=float)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("Transformation matrix must be a finite 4x4 matrix")
        if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
            raise ValueError("Transformation matrix must be homogeneous")
        rotation = value[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError("Only rigid transformations (rotation and translation) are supported")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError("Transformation rotation must have determinant +1")
        return RegionExpression(_transform_node(self._node, value))

    def to_csg(self, compiler, parent_matrix=None, name: str = "region"):
        """Compile the high-level expression to the backend-independent CSG AST.

        The compiler owns the SurfaceRegistry, so geometrically identical
        surfaces from different expressions can be deduplicated. This method
        recursively converts leaves first, then recreates each set operation
        using the same Halfspace/Intersection/UnionRegion/Complement classes
        used by the implicit hierarchy compiler and both simulation exporters.
        """
        from .csg import Complement, Halfspace, Intersection, UnionRegion

        parent = np.eye(4) if parent_matrix is None else np.asarray(parent_matrix, dtype=float)
        if parent.shape != (4, 4):
            raise ValueError("parent_matrix must be a 4x4 matrix")

        leaf_index = [0]

        def compile_node(node: ExpressionNode):
            if isinstance(node, _ShapeLeaf):
                leaf_index[0] += 1
                full_matrix = parent @ node.matrix_array()
                region, _surfaces = compiler.primitive_region(
                    node.shape, full_matrix, f"{name}_operand_{leaf_index[0]}"
                )
                if region is None:
                    raise ValueError(f"Region expression '{name}' contains an empty primitive")
                return region

            compiled = tuple(compile_node(term) for term in node.terms)
            if node.operation == "intersection":
                return Intersection(compiled)
            if node.operation == "union":
                return UnionRegion(compiled)
            if node.operation == "difference":
                # A - B is exactly A intersect complement(B). Keeping this
                # identity explicit makes the region easy to serialize and
                # maps directly to OpenMC and AGORA Boolean operators.
                return Intersection((compiled[0], Complement(compiled[1])))
            if node.operation == "complement":
                return Complement(compiled[0])
            raise ValueError(f"Unsupported Boolean operation: {node.operation}")

        return compile_node(self._node)

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation of the user expression."""
        def serialize(node):
            if isinstance(node, _ShapeLeaf):
                return {
                    "op": "primitive",
                    "primitive": node.shape.analytic(),
                    "transform": node.matrix_array().tolist(),
                }
            return {
                "op": node.operation,
                "terms": [serialize(term) for term in node.terms],
            }
        return serialize(self._node)

    def __repr__(self) -> str:
        return f"RegionExpression({self.to_dict()!r})"


def as_region_expression(value) -> RegionExpression:
    """Normalize a Primitive or RegionExpression into a RegionExpression."""
    if isinstance(value, RegionExpression):
        return value
    if isinstance(value, Primitive):
        return RegionExpression.from_primitive(value)
    raise TypeError(
        "Expected an analytical Primitive or RegionExpression, "
        f"got {type(value).__name__}"
    )


def union_all(regions: Iterable[object]) -> RegionExpression:
    """Build the union of an iterable of primitives/expressions.

    Empty unions are rejected rather than treated as an implicit empty or
    infinite set. This avoids accidentally generating an invalid material cell.
    """
    terms = tuple(as_region_expression(item) for item in regions)
    if not terms:
        raise ValueError("union_all() requires at least one region")
    result = terms[0]
    for term in terms[1:]:
        result = result | term
    return result


def intersection_all(regions: Iterable[object]) -> RegionExpression:
    """Build the intersection of an iterable of primitives/expressions."""
    terms = tuple(as_region_expression(item) for item in regions)
    if not terms:
        raise ValueError("intersection_all() requires at least one region")
    result = terms[0]
    for term in terms[1:]:
        result = result & term
    return result
