"""Exactly preserve the pinned LIBERO glass-blur transform and NumPy RNG stream."""
import ast
import hashlib
import inspect
import numpy as np
from numba import njit

SOURCE_SHA256 = 'b0a3ad46f4f1af7761c39a7576a8997ea9f74e2bb49caa098990d0e6aede8fb5'


@njit(cache=True)
def _apply(x, delta, iterations, offsets):
    k = 0
    for i in range(iterations):
        for h in range(x.shape[0] - delta, delta, -1):
            for w in range(x.shape[1] - delta, delta, -1):
                dx, dy = offsets[k, 0], offsets[k, 1]
                hp, wp = h + dy, w + dx
                # Preserve the reference's NumPy view-aliasing semantics. It
                # is NOT a true value swap: RHS views alias after the first
                # assignment. Replacing it with a conventional swap is wrong.
                for ch in range(x.shape[2]):
                    x[h, w, ch] = x[hp, wp, ch]
                    x[hp, wp, ch] = x[h, w, ch]
                k += 1


def _shuffle(x, delta, iterations):
    n = iterations * max(0, x.shape[0]-2*delta) * max(0, x.shape[1]-2*delta)
    offsets = np.random.randint(-delta, delta, size=(n, 2))
    _apply(x, delta, iterations, offsets)


def install():
    from libero.libero.envs import env_wrapper as module
    original = module.glass_blur
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Unrecognized glass_blur source; refusing optimization')
    tree = ast.parse(source)
    function = tree.body[0]
    loops = [i for i, node in enumerate(function.body) if isinstance(node, ast.For)]
    assert len(loops) == 1
    function.body[loops[0]] = ast.parse('_vla_shuffle(x, c[1], c[2])').body[0]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _vla_shuffle=_shuffle)
    exec(compile(tree, '<pinned-fast-glass-blur>', 'exec'), namespace)
    module.glass_blur = namespace['glass_blur']
    print('[fast-glass-blur] enabled, pinned source='+SOURCE_SHA256, flush=True)
    return original, module.glass_blur
