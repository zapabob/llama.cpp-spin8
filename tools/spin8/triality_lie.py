"""Full 28-parameter infinitesimal Spin(8) related triples.

The convention is A(x*y) = (B*x)*(C*y).  A single parameter vector
theta in R^28 supplies three skew matrices K_A, K_B, K_C. Their matrix
exponentials form a related triple. Three independent parameter vectors
therefore supply nine rotations with 84 intrinsic parameters.

All skew basis matrices have Frobenius norm one. The default octonion
multiplication tensor is loaded from the accompanying reference codec;
callers may supply the tensor explicitly instead. No learned data or
calibration statistics enter the construction.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, expm_frechet


DIM = 8
NCOORD = DIM * (DIM - 1) // 2


def skew_basis() -> np.ndarray:
    """Return the standard Frobenius-orthonormal basis of so(8)."""
    out = np.zeros((NCOORD, DIM, DIM), dtype=np.float64)
    for k, (i, j) in enumerate(zip(*np.triu_indices(DIM, 1))):
        out[k, i, j] = 1.0 / np.sqrt(2.0)
        out[k, j, i] = -1.0 / np.sqrt(2.0)
    return out


def default_multiplication_tensor() -> np.ndarray:
    """Read the reference codec's exact Fano-plane convention."""
    directory = Path(__file__).resolve().parent
    candidates = (directory / "reference_codec.py", directory / "spin8_codec.py",
                  directory.parent / "spin8_quantization" / "spin8_codec.py")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("Pass MULT explicitly or include spin8_g128.py beside the learning package")
    spec = importlib.util.spec_from_file_location("_spin8_lie_reference_codec", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the reference multiplication tensor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return np.asarray(module.MULT, dtype=np.float64).copy()


def _tensor(mult: np.ndarray | None) -> np.ndarray:
    out = default_multiplication_tensor() if mult is None else np.asarray(mult, dtype=np.float64)
    if out.shape != (DIM, DIM, DIM) or not np.isfinite(out).all():
        raise ValueError("MULT must be a finite (8, 8, 8) array with input,input,output indices")
    return out


def build_triality_lie_basis(mult: np.ndarray | None = None) -> np.ndarray:
    """Return shape (3, 28, 8, 8), ordered A, B, C.

    For each standard skew basis matrix A, solve the linearized identity
    A(x*y) = (B*x)*y + x*(C*y) for skew B and C. The 512 scalar equations
    have 56 unknowns and full column rank. Because the vector projection
    of the related-triple Lie algebra is an isomorphism onto so(8), this
    constructs all 28 degrees of freedom, without the restriction of a
    fixed, short product of unit-octonion factors.
    """
    m = _tensor(mult)
    e = skew_basis()
    b_action = np.einsum("api,pjk->aijk", e, m)
    c_action = np.einsum("aqj,iqk->aijk", e, m)
    lhs = np.concatenate((b_action, c_action), axis=0).reshape(2 * NCOORD, -1).T
    targets = np.einsum("akl,ijl->aijk", e, m).reshape(NCOORD, -1).T
    solution, _, rank, _ = np.linalg.lstsq(lhs, targets, rcond=None)
    if rank != 2 * NCOORD:
        raise ValueError("The supplied multiplication tensor has a degenerate triality system")
    residual = np.max(np.abs(lhs @ solution - targets))
    if residual > 5e-12:
        raise ValueError(f"The supplied tensor does not satisfy the required infinitesimal triality identity: {residual}")
    b = np.einsum("pa,pij->aij", solution[:NCOORD], e)
    c = np.einsum("pa,pij->aij", solution[NCOORD:], e)
    return np.stack((e, b, c))


def coefficient_maps(basis: np.ndarray) -> np.ndarray:
    """Return maps T_r with K_r(theta) = sum_p (T_r theta)_p E_p.

    Each result has shape (28,28). All three are orthogonal to numerical
    precision; the first is identity, since the A basis was prescribed.
    """
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != (3, NCOORD, DIM, DIM):
        raise ValueError("Expected basis shape (3,28,8,8)")
    return np.einsum("raij,pij->rpa", basis, skew_basis())


def _parameters_and_base(parameters: np.ndarray, initial_bank: np.ndarray | None):
    theta = np.asarray(parameters, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != NCOORD or not np.isfinite(theta).all():
        raise ValueError("Parameters must be a finite (number_of_triples,28) array")
    if initial_bank is None:
        base = np.broadcast_to(np.eye(DIM), (len(theta), 3, DIM, DIM))
    else:
        base = np.asarray(initial_bank, dtype=np.float64)
        if base.shape == (3 * len(theta), DIM, DIM):
            base = base.reshape(len(theta), 3, DIM, DIM)
        if base.shape != (len(theta), 3, DIM, DIM) or not np.isfinite(base).all():
            raise ValueError("Initial bank must have shape (3*n,8,8) or (n,3,8,8)")
    return theta, base


def parameterized_bank(parameters: np.ndarray, basis: np.ndarray,
                       initial_bank: np.ndarray | None = None) -> np.ndarray:
    """Return exp(K_r(theta)) @ R0_r, flattened to shape (3*n,8,8).

    A triality basis and a related-triple initial bank preserve triality.
    A fixed random-sharing basis of the same shape is also accepted, but
    then only skew symmetry/orthogonality, not triality, is guaranteed.
    Use one fixed coordinate vector through training: repeatedly applying
    independent constrained increments can enlarge an arbitrary sharing
    family's reachable set through Lie brackets.
    """
    theta, base = _parameters_and_base(parameters, initial_bank)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != (3, NCOORD, DIM, DIM):
        raise ValueError("Expected basis shape (3,28,8,8)")
    generators = np.einsum("na,raij->nrij", theta, basis)
    out = np.empty_like(generators)
    for n in range(len(theta)):
        for r in range(3):
            out[n, r] = expm(generators[n, r]) @ base[n, r]
    return out.reshape(-1, DIM, DIM)


def parameter_gradient(parameters: np.ndarray, basis: np.ndarray,
                       bank_gradient: np.ndarray,
                       initial_bank: np.ndarray | None = None) -> np.ndarray:
    """Pull a matrix-bank Frobenius gradient back to shared parameters.

    Uses the adjoint Frechet derivative of exp, not an approximation that
    assumes the matrix generator commutes with its parameter directions.
    The returned array has the same shape as parameters.
    """
    theta, base = _parameters_and_base(parameters, initial_bank)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != (3, NCOORD, DIM, DIM):
        raise ValueError("Expected basis shape (3,28,8,8)")
    cotangent = np.asarray(bank_gradient, dtype=np.float64)
    if cotangent.shape == (3 * len(theta), DIM, DIM):
        cotangent = cotangent.reshape(len(theta), 3, DIM, DIM)
    if cotangent.shape != (len(theta), 3, DIM, DIM):
        raise ValueError("Bank gradient has an incompatible shape")
    generators = np.einsum("na,raij->nrij", theta, basis)
    result = np.zeros_like(theta)
    for n in range(len(theta)):
        for r in range(3):
            exp_gradient = cotangent[n, r] @ base[n, r].T
            generator_gradient = expm_frechet(
                generators[n, r].T, exp_gradient, compute_expm=False
            )
            result[n] += np.einsum("ij,aij->a", generator_gradient, basis[r])
    return result


def diagnostics(mult: np.ndarray | None = None, seed: int = 271828) -> dict[str, float]:
    """Numerical checks of the defining identities and a directional VJP."""
    m = _tensor(mult)
    basis = build_triality_lie_basis(m)
    a, b, c = basis
    infinitesimal = (np.einsum("akl,ijl->aijk", a, m)
                    - np.einsum("api,pjk->aijk", b, m)
                    - np.einsum("aqj,iqk->aijk", c, m))
    maps = coefficient_maps(basis)
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.3, size=(3, NCOORD))
    bank = parameterized_bank(theta, basis).reshape(3, 3, DIM, DIM)
    product_left = np.einsum("nkl,ijl->nijk", bank[:, 0], m)
    product_right = np.einsum("npi,nqj,pqk->nijk", bank[:, 1], bank[:, 2], m)
    cotangent = rng.normal(size=(9, DIM, DIM))
    # Nonidentity base verifies the multiplication-order contribution to VJP.
    base = parameterized_bank(rng.normal(scale=0.3, size=theta.shape), basis)
    gradient = parameter_gradient(theta, basis, cotangent, base)
    direction = rng.normal(size=theta.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = parameterized_bank(theta + eps * direction, basis, base)
    minus = parameterized_bank(theta - eps * direction, basis, base)
    finite_difference = np.sum((plus - minus) * cotangent) / (2 * eps)
    analytic = float(np.sum(gradient * direction))
    result = {
        "basis_skew_max_abs": float(np.max(np.abs(basis + basis.swapaxes(-1, -2)))),
        "infinitesimal_identity_max_abs": float(np.max(np.abs(infinitesimal))),
        "coefficient_map_orthogonality_max_abs": float(np.max(np.abs(maps @ maps.swapaxes(-1, -2) - np.eye(NCOORD)))),
        "exp_triality_identity_max_abs": float(np.max(np.abs(product_left - product_right))),
        "exp_orthogonality_max_abs": float(np.max(np.abs(bank @ bank.swapaxes(-1, -2) - np.eye(DIM)))),
        "gradient_directional_abs_error": float(abs(finite_difference - analytic)),
        "gradient_directional_relative_error": float(abs(finite_difference - analytic) / max(1.0, abs(finite_difference), abs(analytic))),
    }
    for key, value in result.items():
        tolerance = 1e-7 if "gradient" in key else 1e-11
        if value > tolerance:
            raise AssertionError(f"{key}={value} exceeds {tolerance}")
    return result


if __name__ == "__main__":
    print(json.dumps(diagnostics(), indent=2))
