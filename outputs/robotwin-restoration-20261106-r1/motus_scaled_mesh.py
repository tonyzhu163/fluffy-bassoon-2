"""Opt-in, source-bound SAPIEN mesh-to-FCL scale conversion.

Retain the installed converter's shape types, poses, faces and collision checks.
Only bake convex/triangle mesh scale into vertices. Never edit the shared SDK.
"""
import ast
import hashlib
import inspect
import textwrap


def scaled_converter(original, expected_source_sha256):
    source = textwrap.dedent(inspect.getsource(original))
    assert len(expected_source_sha256) == 64
    assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest() == expected_source_sha256
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    function = tree.body[0]
    assert len(function.decorator_list) == 1 and isinstance(function.decorator_list[0], ast.Name)
    assert function.decorator_list[0].id == 'staticmethod'
    function.decorator_list = []

    class ScaleMeshes(ast.NodeTransformer):
        removed_asserts = 0
        scaled_vertices = 0

        def visit_Assert(self, node):
            if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and 'Not unit scale' in n.value
                   for n in ast.walk(node)):
                self.removed_asserts += 1
                return None
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'shape' and node.attr == 'vertices':
                self.scaled_vertices += 1
                return ast.copy_location(ast.BinOp(left=node, op=ast.Mult(),
                    right=ast.Attribute(value=ast.Name(id='shape', ctx=ast.Load()), attr='scale', ctx=ast.Load())), node)
            return self.generic_visit(node)

    edit = ScaleMeshes(); tree = edit.visit(tree)
    assert edit.removed_asserts == 1 and edit.scaled_vertices == 2
    ast.fix_missing_locations(tree)
    namespace = original.__globals__.copy()
    exec(compile(tree, '<source-bound-scaled-mesh-converter>', 'exec'), namespace)
    return namespace[function.name]


def install(expected_source_sha256):
    from mplib.sapien_utils import SapienPlanningWorld
    original = SapienPlanningWorld.convert_physx_component
    fixed = scaled_converter(original, expected_source_sha256)
    SapienPlanningWorld.convert_physx_component = staticmethod(fixed)
