"""constrained crystallographic phase alignment"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Final, Optional, Union

import gemmi
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len
from scipy.optimize import OptimizeResult, minimize

from reciprocalspaceship.decorators import spacegroupify
from reciprocalspaceship.utils.phases import canonicalize_phases

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


FloatArray: TypeAlias = NDArray[np.float64]
IntegerArray: TypeAlias = NDArray[np.int64]
SpaceGroupLike: TypeAlias = Union[str, int, gemmi.SpaceGroup]

NUMBER_OF_CRYSTALLOGRAPHIC_AXES: Final[int] = 3
FULL_ROTATION_DEGREES: Final[float] = 360.0
FULL_ROTATION_RADIANS: Final[float] = float(2.0 * np.pi)
ORIGIN_DENOMINATOR: Final[int] = int(gemmi.Op.DEN)
SINGULAR_VALUE_TOLERANCE: Final[float] = 1e-10
OPTIMIZER_GRADIENT_TOLERANCE: Final[float] = 1e-7
OPTIMIZER_ACCEPTABLE_GRADIENT: Final[float] = 1e-5
OPTIMIZER_MAXIMUM_ITERATIONS: Final[int] = 500
TRANSLATION_ROUNDING_DECIMALS: Final[int] = 12
PERIODIC_OFFSETS: Final[FloatArray] = np.asarray(
    tuple(product((-1.0, 0.0, 1.0), repeat=NUMBER_OF_CRYSTALLOGRAPHIC_AXES)),
    dtype=np.float64,
)


class PhaseAlignmentInputError(ValueError):
    """Raised when phase-alignment inputs are invalid or insufficient."""


class PhaseAlignmentOptimizationError(RuntimeError):
    """Raised when continuous phase alignment does not converge."""


def _validate_miller_indices(miller_indices: ArrayLike) -> IntegerArray:
    miller_indices_array = np.asarray(miller_indices)
    if (
        miller_indices_array.ndim != 2
        or miller_indices_array.shape[1] != NUMBER_OF_CRYSTALLOGRAPHIC_AXES
    ):
        msg = f"miller_indices must have shape (n, 3); got {miller_indices_array.shape}"
        raise PhaseAlignmentInputError(msg)
    try:
        floating_miller_indices = np.asarray(miller_indices_array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "miller_indices must contain integer-valued numbers"
        raise PhaseAlignmentInputError(msg) from error
    if not np.isfinite(floating_miller_indices).all():
        msg = "miller_indices must contain only finite values"
        raise PhaseAlignmentInputError(msg)
    if not np.equal(floating_miller_indices, np.rint(floating_miller_indices)).all():
        msg = "miller_indices must contain only integer-valued entries"
        raise PhaseAlignmentInputError(msg)
    return np.asarray(floating_miller_indices, dtype=np.int64)


def _validate_phases(phases: ArrayLike, *, name: str) -> FloatArray:
    try:
        phases_array = np.asarray(phases, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} must contain numeric values"
        raise PhaseAlignmentInputError(msg) from error
    if phases_array.ndim != 1:
        msg = f"{name} must have shape (n,); got {phases_array.shape}"
        raise PhaseAlignmentInputError(msg)
    if not np.isfinite(phases_array).all():
        msg = f"{name} must contain only finite values"
        raise PhaseAlignmentInputError(msg)
    return phases_array


def _validate_weights(
    weights: Optional[ArrayLike],
    *,
    number_of_phases: int,
) -> FloatArray:
    if weights is None:
        return np.ones(number_of_phases, dtype=np.float64)
    try:
        weights_array = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "weights must contain numeric values"
        raise PhaseAlignmentInputError(msg) from error
    if weights_array.shape != (number_of_phases,):
        msg = (
            f"weights must have shape ({number_of_phases},); got {weights_array.shape}"
        )
        raise PhaseAlignmentInputError(msg)
    if not np.isfinite(weights_array).all():
        msg = "weights must contain only finite values"
        raise PhaseAlignmentInputError(msg)
    if (weights_array < 0.0).any():
        msg = "weights must be nonnegative"
        raise PhaseAlignmentInputError(msg)
    if not (weights_array > 0.0).any():
        msg = "at least one weight must be positive"
        raise PhaseAlignmentInputError(msg)
    return weights_array


def _rotation_constraints(spacegroup: gemmi.SpaceGroup) -> IntegerArray:
    identity_rotation = np.eye(NUMBER_OF_CRYSTALLOGRAPHIC_AXES, dtype=np.int64)
    constraints = [
        identity_rotation
        - np.asarray(operation.rot, dtype=np.int64) // ORIGIN_DENOMINATOR
        for operation in spacegroup.operations().sym_ops
    ]
    return np.concatenate(constraints, axis=0)


def _polar_basis(rotation_constraints: IntegerArray) -> FloatArray:
    _, singular_values, right_singular_vectors = np.linalg.svd(
        rotation_constraints.astype(np.float64),
        full_matrices=True,
    )
    rank = int(np.count_nonzero(singular_values > SINGULAR_VALUE_TOLERANCE))
    return np.asarray(right_singular_vectors[rank:].T, dtype=np.float64)


def _allowed_grid_origins(
    spacegroup: gemmi.SpaceGroup,
    rotation_constraints: IntegerArray,
) -> FloatArray:
    grid_coordinates = (
        np.indices(
            (ORIGIN_DENOMINATOR,) * NUMBER_OF_CRYSTALLOGRAPHIC_AXES,
            dtype=np.int64,
        )
        .reshape(NUMBER_OF_CRYSTALLOGRAPHIC_AXES, -1)
        .T
    )
    centering_translations = (
        np.asarray(spacegroup.operations().cen_ops, dtype=np.int64) % ORIGIN_DENOMINATOR
    )
    allowed = np.ones(len(grid_coordinates), dtype=np.bool_)
    for constraint in rotation_constraints.reshape(
        -1,
        NUMBER_OF_CRYSTALLOGRAPHIC_AXES,
        NUMBER_OF_CRYSTALLOGRAPHIC_AXES,
    ):
        origin_shifts = grid_coordinates @ constraint.T % ORIGIN_DENOMINATOR
        allowed &= np.all(
            origin_shifts[:, None, :] == centering_translations[None, :, :],
            axis=2,
        ).any(axis=1)
    return np.asarray(
        grid_coordinates[allowed] / ORIGIN_DENOMINATOR,
        dtype=np.float64,
    )


def _phase_loss(
    fractional_translation: FloatArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> float:
    residuals = (
        phase_differences
        + FULL_ROTATION_RADIANS * miller_indices @ fractional_translation
    )
    return float(normalized_weights @ (1.0 - np.cos(residuals)))


def _coarse_fractional_translation(
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> FloatArray:
    maximum_indices = np.max(np.abs(miller_indices), axis=0)
    grid_shape = tuple(
        next_fast_len(int(2 * maximum_index + 1)) for maximum_index in maximum_indices
    )
    fourier_coefficients = np.zeros(grid_shape, dtype=np.complex128)
    wrapped_indices = tuple(
        miller_indices[:, axis] % grid_shape[axis]
        for axis in range(NUMBER_OF_CRYSTALLOGRAPHIC_AXES)
    )
    np.add.at(
        fourier_coefficients,
        wrapped_indices,
        normalized_weights * np.exp(1j * phase_differences),
    )
    correlation = np.fft.ifftn(fourier_coefficients).real
    peak_index = np.asarray(
        np.unravel_index(int(np.argmax(correlation)), grid_shape),
        dtype=np.float64,
    )
    return peak_index / np.asarray(grid_shape, dtype=np.float64)


def _candidate_translations(
    allowed_grid_origins: FloatArray,
    polar_basis: FloatArray,
    coarse_translation: FloatArray,
) -> FloatArray:
    polar_projection = polar_basis @ polar_basis.T
    periodic_grid_origins = (
        allowed_grid_origins[:, None, :] + PERIODIC_OFFSETS[None, :, :]
    )
    differences = coarse_translation - periodic_grid_origins
    polar_components = differences @ polar_projection
    nonpolar_distances = np.sum((differences - polar_components) ** 2, axis=2)
    closest_periodic_images = np.argmin(nonpolar_distances, axis=1)
    row_indices = np.arange(len(allowed_grid_origins))
    translations = (
        periodic_grid_origins[row_indices, closest_periodic_images]
        + polar_components[row_indices, closest_periodic_images]
    ) % 1.0
    rounded_translations = np.round(
        translations, decimals=TRANSLATION_ROUNDING_DECIMALS
    )
    return np.unique(rounded_translations, axis=0)


def _refine_translation(
    starting_translation: FloatArray,
    polar_basis: FloatArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> tuple[FloatArray, float]:
    def objective(polar_coordinates: FloatArray) -> tuple[float, FloatArray]:
        translation = starting_translation + polar_basis @ polar_coordinates
        residuals = (
            phase_differences + FULL_ROTATION_RADIANS * miller_indices @ translation
        )
        weighted_sines = normalized_weights * np.sin(residuals)
        loss = float(normalized_weights @ (1.0 - np.cos(residuals)))
        translation_gradient = FULL_ROTATION_RADIANS * miller_indices.T @ weighted_sines
        polar_gradient = polar_basis.T @ translation_gradient
        return loss, np.asarray(polar_gradient, dtype=np.float64)

    result: OptimizeResult = minimize(
        objective,
        np.zeros(polar_basis.shape[1], dtype=np.float64),
        method="BFGS",
        jac=True,
        options={
            "gtol": OPTIMIZER_GRADIENT_TOLERANCE,
            "maxiter": OPTIMIZER_MAXIMUM_ITERATIONS,
        },
    )
    result_gradient = np.asarray(result.jac, dtype=np.float64)
    acceptable_precision_limited_result = (
        np.isfinite(result.fun)
        and np.isfinite(result_gradient).all()
        and np.linalg.norm(result_gradient, ord=np.inf) <= OPTIMIZER_ACCEPTABLE_GRADIENT
    )
    if not result.success and not acceptable_precision_limited_result:
        msg = f"phase alignment did not converge: {result.message}"
        raise PhaseAlignmentOptimizationError(msg)
    translation = (
        starting_translation + polar_basis @ np.asarray(result.x, dtype=np.float64)
    ) % 1.0
    return np.asarray(translation, dtype=np.float64), float(result.fun)


def _continuous_translation(
    allowed_grid_origins: FloatArray,
    polar_basis: FloatArray,
    miller_indices: IntegerArray,
    phase_differences: FloatArray,
    normalized_weights: FloatArray,
) -> FloatArray:
    positive_weight_indices = miller_indices[normalized_weights > 0.0]
    identifiable_rank = np.linalg.matrix_rank(
        positive_weight_indices @ polar_basis,
        tol=SINGULAR_VALUE_TOLERANCE,
    )
    if identifiable_rank != polar_basis.shape[1]:
        msg = "positive-weight reflections do not identify every continuous origin direction"
        raise PhaseAlignmentInputError(msg)

    coarse_translation = _coarse_fractional_translation(
        miller_indices,
        phase_differences,
        normalized_weights,
    )
    if polar_basis.shape[1] == NUMBER_OF_CRYSTALLOGRAPHIC_AXES:
        starting_translations = coarse_translation[None, :]
    else:
        starting_translations = _candidate_translations(
            allowed_grid_origins,
            polar_basis,
            coarse_translation,
        )

    refined_translations = [
        _refine_translation(
            starting_translation,
            polar_basis,
            miller_indices,
            phase_differences,
            normalized_weights,
        )
        for starting_translation in starting_translations
    ]
    best_translation, _ = min(refined_translations, key=lambda result: result[1])
    return best_translation


@spacegroupify
def align_phases(
    miller_indices: ArrayLike,
    phases: ArrayLike,
    reference_phases: ArrayLike,
    spacegroup: SpaceGroupLike,
    *,
    weights: Optional[ArrayLike] = None,
) -> tuple[FloatArray, FloatArray]:
    """Align phases using only origin shifts allowed by the space group.

    Parameters
    ----------
    miller_indices : array-like
        Miller indices with shape ``(n, 3)``.
    phases : array-like
        Phases to align, in degrees.
    reference_phases : array-like
        Reference phases, in degrees.
    spacegroup : str, int, gemmi.SpaceGroup
        Space group defining the allowed origin shifts.
    weights : array-like, optional
        Nonnegative reflection weights. Zero-weight reflections are ignored.

    Returns
    -------
    aligned_phases : numpy.ndarray
        A new array of phases in the interval ``[-180, 180)``.
    fractional_translation : numpy.ndarray
        Fractional translation added to ``phases`` to produce ``aligned_phases``.

    Raises
    ------
    PhaseAlignmentInputError
        If the inputs are invalid or do not identify every continuous origin direction.
    PhaseAlignmentOptimizationError
        If optimization along a continuous origin direction does not converge.

    Notes
    -----
    The objective is the weighted circular phase residual. Nonpolar origin choices are
    enumerated exactly from Gemmi symmetry operations. Polar origin directions are refined
    continuously while all perpendicular components remain on an allowed origin manifold.
    """
    validated_miller_indices = _validate_miller_indices(miller_indices)
    validated_phases = _validate_phases(phases, name="phases")
    validated_reference_phases = _validate_phases(
        reference_phases,
        name="reference_phases",
    )
    number_of_reflections = len(validated_miller_indices)
    if len(validated_phases) != number_of_reflections:
        msg = "miller_indices and phases must have the same length"
        raise PhaseAlignmentInputError(msg)
    if len(validated_reference_phases) != number_of_reflections:
        msg = "miller_indices and reference_phases must have the same length"
        raise PhaseAlignmentInputError(msg)
    if number_of_reflections == 0:
        msg = "at least one reflection is required"
        raise PhaseAlignmentInputError(msg)

    validated_weights = _validate_weights(
        weights,
        number_of_phases=number_of_reflections,
    )
    normalized_weights = validated_weights / np.sum(validated_weights)
    if not isinstance(spacegroup, gemmi.SpaceGroup):
        msg = (
            f"spacegroup could not be converted to gemmi.SpaceGroup; got {spacegroup!r}"
        )
        raise PhaseAlignmentInputError(msg)
    validated_spacegroup = spacegroup
    rotation_constraints = _rotation_constraints(validated_spacegroup)
    polar_basis = _polar_basis(rotation_constraints)
    allowed_grid_origins = _allowed_grid_origins(
        validated_spacegroup,
        rotation_constraints,
    )
    phase_differences = np.deg2rad(
        canonicalize_phases(validated_phases - validated_reference_phases),
    )

    if polar_basis.shape[1] == 0:
        losses = np.asarray(
            [
                _phase_loss(
                    origin,
                    validated_miller_indices,
                    phase_differences,
                    normalized_weights,
                )
                for origin in allowed_grid_origins
            ],
            dtype=np.float64,
        )
        fractional_translation = allowed_grid_origins[int(np.argmin(losses))]
    else:
        fractional_translation = _continuous_translation(
            allowed_grid_origins,
            polar_basis,
            validated_miller_indices,
            phase_differences,
            normalized_weights,
        )

    aligned_phases = canonicalize_phases(
        validated_phases
        + FULL_ROTATION_DEGREES * validated_miller_indices @ fractional_translation
    )
    return (
        np.asarray(aligned_phases, dtype=np.float64),
        np.asarray(fractional_translation, dtype=np.float64),
    )
