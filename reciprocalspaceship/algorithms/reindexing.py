"""correlation-based crystallographic reindexing"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Optional
from warnings import warn

import gemmi
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from reciprocalspaceship.dataset import DataSet
from reciprocalspaceship.dtypes import (
    AnomalousDifferenceDtype,
    FriedelIntensityDtype,
    FriedelStructureFactorAmplitudeDtype,
    HendricksonLattmanDtype,
    IntensityDtype,
    NormalizedStructureFactorAmplitudeDtype,
    StandardDeviationFriedelIDtype,
    StandardDeviationFriedelSFDtype,
    StructureFactorAmplitudeDtype,
)

from ._errors import (
    LowCorrelationWarning,
    NoClearSolutionError,
    PhaseAlignmentInputError,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

DEFAULT_MAXIMUM_OBLIQUITY: Final[float] = 1e-6
# Conservative gates calibrated on seeded amplitude-noise trials with 6OVT.
DEFAULT_REINDEXING_WARNING_CORRELATION: Final[float] = 0.5
DEFAULT_REINDEXING_MINIMUM_CORRELATION: Final[float] = 0.2
DEFAULT_REINDEXING_MINIMUM_CORRELATION_GAP: Final[float] = 0.05
MINIMUM_COMMON_REFLECTIONS: Final[int] = 3
TARGET_REFLECTIONS_PER_RESOLUTION_BIN: Final[int] = 100
MAXIMUM_RESOLUTION_BINS: Final[int] = 20
MAXIMUM_CORRELATION_GAP: Final[float] = 2.0
IDENTITY_OPERATION: Final[str] = "x,y,z"
ANOMALOUS_DTYPES: Final[tuple[type[object], ...]] = (
    AnomalousDifferenceDtype,
    FriedelIntensityDtype,
    FriedelStructureFactorAmplitudeDtype,
    StandardDeviationFriedelIDtype,
    StandardDeviationFriedelSFDtype,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ReindexingCandidate:
    """A candidate indexing operation and its correlation score.

    Attributes
    ----------
    operation : gemmi.Op
        Proper lattice-ambiguity representative applied to the moving Miller indices.
    correlation : float
        Signed Pearson correlation between resolution-normalized intensities.
    number_of_reflections : int
        Number of finite reflections shared by every candidate and the reference.

    """

    operation: gemmi.Op
    correlation: float
    number_of_reflections: int


@dataclass(frozen=True)
class ReindexingResult:
    """The selected indexing operation and its ranked alternatives.

    Attributes
    ----------
    dataset : DataSet
        Reindexed copy in the reference space group, cell, and reciprocal ASU. Its
        phase origin may still differ from the reference by a symmetry-allowed shift.
    operation : gemmi.Op
        Lattice-ambiguity representative applied to the moving Miller indices to
        obtain ``dataset``.
    correlation : float
        Correlation of the selected operation.
    runner_up_correlation : float, optional
        Correlation of the next-best operation, if one exists.
    correlation_gap : float, optional
        Best-minus-runner-up correlation, if a runner-up exists.
    candidates : tuple of ReindexingCandidate
        All candidates, sorted from highest to lowest correlation.

    """

    dataset: DataSet
    operation: gemmi.Op
    correlation: float
    runner_up_correlation: Optional[float]
    correlation_gap: Optional[float]
    candidates: tuple[ReindexingCandidate, ...]


def _validate_maximum_obliquity(max_obliquity: float) -> float:
    try:
        validated_maximum_obliquity = float(max_obliquity)
    except (TypeError, ValueError) as error:
        msg = f"max_obliquity must be numeric; got {max_obliquity!r}"
        raise ValueError(msg) from error
    if (
        not np.isfinite(validated_maximum_obliquity)
        or validated_maximum_obliquity < 0.0
    ):
        msg = f"max_obliquity must be finite and nonnegative; got {max_obliquity!r}"
        raise ValueError(msg)
    return validated_maximum_obliquity


def _validate_symmetry_metadata(dataset: DataSet, *, name: str) -> None:
    if not isinstance(dataset.spacegroup, gemmi.SpaceGroup):
        msg = f"{name}.spacegroup must be set"
        raise PhaseAlignmentInputError(msg)
    if not isinstance(dataset.cell, gemmi.UnitCell):
        msg = f"{name}.cell must be set"
        raise PhaseAlignmentInputError(msg)


def has_reindexing_ambiguity(
    dataset: DataSet,
    *,
    max_obliquity: float = DEFAULT_MAXIMUM_OBLIQUITY,
) -> bool:
    """Test whether a dataset admits an alternative indexing operation.

    Parameters
    ----------
    dataset : DataSet
        Dataset supplying both the space group and unit-cell metric.
    max_obliquity : float, optional
        Maximum lattice-symmetry obliquity in degrees. The default includes only
        exact merohedral ambiguities; larger values include pseudo-merohedry.

    Returns
    -------
    bool
        Whether Gemmi finds at least one nonredundant proper reindexing operation.

    Raises
    ------
    ValueError
        If crystallographic metadata or ``max_obliquity`` is invalid.

    Notes
    -----
    Indexing ambiguity depends on the unit-cell metric as well as the space group.
    Hand inversion is deliberately excluded, matching Gemmi and Pointless.
    """
    if not isinstance(dataset, DataSet):
        msg = f"dataset must be an rs.DataSet; got {type(dataset).__name__}"
        raise ValueError(msg)
    _validate_symmetry_metadata(dataset, name="dataset")
    validated_maximum_obliquity = _validate_maximum_obliquity(max_obliquity)
    return bool(
        gemmi.find_twin_laws(
            dataset.cell,
            dataset.spacegroup,
            max_obliq=validated_maximum_obliquity,
            all_ops=False,
        )
    )


def _validate_dataset(dataset: DataSet, *, data_key: str, name: str) -> None:
    if not isinstance(dataset, DataSet):
        msg = f"{name} must be an rs.DataSet; got {type(dataset).__name__}"
        raise PhaseAlignmentInputError(msg)
    _validate_symmetry_metadata(dataset, name=name)
    if dataset.merged is not True:
        msg = f"{name} must be a merged DataSet"
        raise PhaseAlignmentInputError(msg)
    anomalous_keys = [
        key for key in dataset if isinstance(dataset.dtypes[key], ANOMALOUS_DTYPES)
    ]
    if anomalous_keys:
        msg = f"{name} contains unsupported anomalous columns: {anomalous_keys}"
        raise PhaseAlignmentInputError(msg)
    hendrickson_lattman_keys = [
        key
        for key in dataset
        if isinstance(dataset.dtypes[key], HendricksonLattmanDtype)
    ]
    if hendrickson_lattman_keys:
        msg = (
            f"{name} contains unsupported Hendrickson-Lattman columns: "
            f"{hendrickson_lattman_keys}"
        )
        raise PhaseAlignmentInputError(msg)
    if data_key not in dataset:
        msg = f"{name} does not contain data key {data_key!r}"
        raise PhaseAlignmentInputError(msg)
    valid_dtypes = (
        StructureFactorAmplitudeDtype,
        NormalizedStructureFactorAmplitudeDtype,
        IntensityDtype,
    )
    if not isinstance(dataset.dtypes[data_key], valid_dtypes):
        msg = f"{name}[{data_key!r}] must have an amplitude or intensity MTZ dtype"
        raise PhaseAlignmentInputError(msg)
    try:
        raw_miller_indices = dataset.reset_index()[["H", "K", "L"]].to_numpy()
    except (KeyError, ValueError) as error:
        msg = f"{name} must contain Miller indices H, K, and L"
        raise PhaseAlignmentInputError(msg) from error
    try:
        floating_miller_indices = np.asarray(raw_miller_indices, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{name} Miller indices must be numeric"
        raise PhaseAlignmentInputError(msg) from error
    if not np.isfinite(floating_miller_indices).all():
        msg = f"{name} Miller indices must be finite"
        raise PhaseAlignmentInputError(msg)
    if not np.equal(floating_miller_indices, np.rint(floating_miller_indices)).all():
        msg = f"{name} Miller indices must be integer-valued"
        raise PhaseAlignmentInputError(msg)
    miller_indices = np.asarray(floating_miller_indices, dtype=np.int64)
    if len(np.unique(miller_indices, axis=0)) != len(miller_indices):
        msg = f"{name} must contain unique Miller indices"
        raise PhaseAlignmentInputError(msg)


def _validate_correlation_threshold(value: float, *, name: str) -> float:
    try:
        validated_value = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be numeric; got {value!r}"
        raise PhaseAlignmentInputError(msg) from error
    if not np.isfinite(validated_value) or not -1.0 <= validated_value <= 1.0:
        msg = f"{name} must be finite and between -1 and 1; got {value!r}"
        raise PhaseAlignmentInputError(msg)
    return validated_value


def _validate_correlation_gap(value: float, *, name: str) -> float:
    try:
        validated_value = float(value)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be numeric; got {value!r}"
        raise PhaseAlignmentInputError(msg) from error
    if (
        not np.isfinite(validated_value)
        or not 0.0 <= validated_value <= MAXIMUM_CORRELATION_GAP
    ):
        msg = (
            f"{name} must be finite and between 0 and "
            f"{MAXIMUM_CORRELATION_GAP:g}; got {value!r}"
        )
        raise PhaseAlignmentInputError(msg)
    return validated_value


def _validate_scoring_thresholds(
    *,
    warning_correlation: float,
    minimum_correlation: float,
    minimum_correlation_gap: float,
    name_prefix: str = "",
) -> tuple[float, float, float]:
    validated_warning_correlation = _validate_correlation_threshold(
        warning_correlation,
        name=f"{name_prefix}warning_correlation",
    )
    validated_minimum_correlation = _validate_correlation_threshold(
        minimum_correlation,
        name=f"{name_prefix}minimum_correlation",
    )
    validated_minimum_correlation_gap = _validate_correlation_gap(
        minimum_correlation_gap,
        name=f"{name_prefix}minimum_correlation_gap",
    )
    return (
        validated_warning_correlation,
        validated_minimum_correlation,
        validated_minimum_correlation_gap,
    )


def _as_asu(dataset: DataSet, operation: gemmi.Op) -> DataSet:
    had_m_isym = "M/ISYM" in dataset
    transformed = dataset.apply_symop(operation).hkl_to_asu()
    if not had_m_isym and "M/ISYM" in transformed:
        transformed.drop(columns="M/ISYM", inplace=True)
    miller_indices = transformed.get_hkls()
    if len(np.unique(miller_indices, axis=0)) != len(miller_indices):
        msg = "merged data must contain unique Miller indices after mapping to the ASU"
        raise PhaseAlignmentInputError(msg)
    return transformed


def _indexed_series(dataset: DataSet, *, data_key: str) -> pd.Series[float]:
    miller_indices = dataset.get_hkls()
    index = pd.MultiIndex.from_arrays(
        miller_indices.T,
        names=("H", "K", "L"),
    )
    return pd.Series(
        dataset[data_key].to_numpy(dtype=np.float64),
        index=index,
        dtype=np.float64,
    )


def _common_finite_index(
    reference_values: pd.Series[float],
    candidate_values: tuple[pd.Series[float], ...],
) -> pd.MultiIndex:
    common_index = reference_values.index
    for values in candidate_values:
        common_index = common_index.intersection(values.index, sort=False)
    finite = np.isfinite(reference_values.loc[common_index].to_numpy(dtype=np.float64))
    for values in candidate_values:
        finite &= np.isfinite(values.loc[common_index].to_numpy(dtype=np.float64))
    return common_index[finite]


def _as_intensities(values: FloatArray, *, amplitude: bool) -> FloatArray:
    if amplitude:
        return np.asarray(values**2, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _resolution_normalize(
    intensities: FloatArray,
    inverse_d_squared: FloatArray,
) -> FloatArray:
    number_of_bins = min(
        MAXIMUM_RESOLUTION_BINS,
        max(1, len(intensities) // TARGET_REFLECTIONS_PER_RESOLUTION_BIN),
    )
    resolution_order = np.argsort(inverse_d_squared)
    normalized = np.empty_like(intensities, dtype=np.float64)
    for bin_indices in np.array_split(resolution_order, number_of_bins):
        scale = float(np.mean(intensities[bin_indices]))
        if not np.isfinite(scale) or scale <= 0.0:
            msg = "intensities must have a positive finite mean in every resolution bin"
            raise PhaseAlignmentInputError(msg)
        normalized[bin_indices] = intensities[bin_indices] / scale
    return normalized


def _pearson_correlation(first: FloatArray, second: FloatArray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2)))
    if not np.isfinite(denominator) or denominator == 0.0:
        msg = "correlation is undefined for constant or nonfinite data"
        raise PhaseAlignmentInputError(msg)
    return float(first_centered @ second_centered / denominator)


def _score_reindexing_candidates(
    dataset: DataSet,
    reference: DataSet,
    operations: tuple[gemmi.Op, ...],
    *,
    data_key: str,
    reference_key: str,
    minimum_correlation: float,
    minimum_correlation_gap: float,
) -> ReindexingResult:
    reference_asu = _as_asu(reference, gemmi.Op(IDENTITY_OPERATION))
    transformed_datasets = tuple(
        _as_asu(dataset, operation) for operation in operations
    )
    reference_values = _indexed_series(reference_asu, data_key=reference_key)
    candidate_values = tuple(
        _indexed_series(transformed, data_key=data_key)
        for transformed in transformed_datasets
    )
    common_index = _common_finite_index(reference_values, candidate_values)
    if len(common_index) < MINIMUM_COMMON_REFLECTIONS:
        msg = (
            "reindexing candidates and reference must share at least "
            f"{MINIMUM_COMMON_REFLECTIONS} finite reflections"
        )
        raise PhaseAlignmentInputError(msg)

    common_miller_indices = np.asarray(common_index.tolist(), dtype=np.int32)
    d_spacings = reference.cell.calculate_d_array(common_miller_indices)
    inverse_d_squared = np.asarray(d_spacings, dtype=np.float64) ** -2.0
    reference_is_amplitude = isinstance(
        reference.dtypes[reference_key],
        (StructureFactorAmplitudeDtype, NormalizedStructureFactorAmplitudeDtype),
    )
    moving_is_amplitude = isinstance(
        dataset.dtypes[data_key],
        (StructureFactorAmplitudeDtype, NormalizedStructureFactorAmplitudeDtype),
    )
    reference_intensities = _as_intensities(
        reference_values.loc[common_index].to_numpy(dtype=np.float64),
        amplitude=reference_is_amplitude,
    )
    normalized_reference = _resolution_normalize(
        reference_intensities,
        inverse_d_squared,
    )

    unsorted_candidates: list[ReindexingCandidate] = []
    for operation_index, operation in enumerate(operations):
        values = candidate_values[operation_index]
        moving_intensities = _as_intensities(
            values.loc[common_index].to_numpy(dtype=np.float64),
            amplitude=moving_is_amplitude,
        )
        normalized_moving = _resolution_normalize(
            moving_intensities,
            inverse_d_squared,
        )
        unsorted_candidates.append(
            ReindexingCandidate(
                operation=operation,
                correlation=_pearson_correlation(
                    normalized_moving,
                    normalized_reference,
                ),
                number_of_reflections=len(common_index),
            )
        )
    candidates = tuple(
        sorted(
            unsorted_candidates,
            key=lambda candidate: candidate.correlation,
            reverse=True,
        )
    )
    best = candidates[0]
    runner_up_correlation = candidates[1].correlation if len(candidates) > 1 else None
    correlation_gap = (
        best.correlation - runner_up_correlation
        if runner_up_correlation is not None
        else None
    )
    if best.correlation < minimum_correlation:
        msg = (
            f"no clear reindexing solution: best correlation {best.correlation:.3f} "
            f"is below {minimum_correlation:.3f}"
        )
        raise NoClearSolutionError(msg)
    if correlation_gap is not None and correlation_gap < minimum_correlation_gap:
        msg = (
            f"no clear reindexing solution: correlation gap {correlation_gap:.3f} "
            f"is below {minimum_correlation_gap:.3f}"
        )
        raise NoClearSolutionError(msg)
    best_index = operations.index(best.operation)
    aligned_dataset = transformed_datasets[best_index]
    aligned_dataset.spacegroup = reference.spacegroup
    aligned_dataset.cell = reference.cell
    return ReindexingResult(
        dataset=aligned_dataset,
        operation=best.operation,
        correlation=best.correlation,
        runner_up_correlation=runner_up_correlation,
        correlation_gap=correlation_gap,
        candidates=candidates,
    )


def reindex_by_correlation(
    dataset: DataSet,
    reference: DataSet,
    *,
    data_key: str,
    reference_key: str,
    max_obliquity: float = DEFAULT_MAXIMUM_OBLIQUITY,
    warning_correlation: float = DEFAULT_REINDEXING_WARNING_CORRELATION,
    minimum_correlation: float = DEFAULT_REINDEXING_MINIMUM_CORRELATION,
    minimum_correlation_gap: float = DEFAULT_REINDEXING_MINIMUM_CORRELATION_GAP,
) -> ReindexingResult:
    """Select the indexing operation with the highest normalized-data correlation.

    Parameters
    ----------
    dataset : DataSet
        Merged moving dataset to reindex.
    reference : DataSet
        Merged reference dataset defining the target indexing.
    data_key : str
        Moving amplitude or intensity column.
    reference_key : str
        Reference amplitude or intensity column.
    max_obliquity : float, optional
        Maximum obliquity in degrees for pseudo-merohedral candidates.
    warning_correlation : float, optional
        Warn when the selected correlation is below this value.
    minimum_correlation : float, optional
        Reject solutions below this correlation.
    minimum_correlation_gap : float, optional
        Reject solutions whose best-minus-runner-up correlation is smaller.

    Returns
    -------
    ReindexingResult
        A reindexed copy and ranked candidate diagnostics.

    Raises
    ------
    PhaseAlignmentInputError
        If inputs are invalid or the dataset has no alternative indexing.
    NoClearSolutionError
        If the scores do not identify a reliable, unique operation.

    Notes
    -----
    Amplitudes are squared and both datasets are normalized by resolution before
    computing signed Pearson correlations, following the strategy used by Pointless.
    The reported operation is applied to ``dataset`` to match ``reference``. Gemmi
    returns one representative of each indexing coset; an equivalent representative
    can differ from Pointless by a space-group operation and an allowed origin shift.
    :func:`~reciprocalspaceship.algorithms.align_phases` resolves that residual shift.
    Only proper reindexing laws are tested; hand inversion is deliberately excluded.
    """
    _validate_dataset(dataset, data_key=data_key, name="dataset")
    _validate_dataset(reference, data_key=reference_key, name="reference")
    if not dataset.is_isomorphous(reference):
        msg = "dataset and reference must be isomorphous"
        raise PhaseAlignmentInputError(msg)
    validated_maximum_obliquity = _validate_maximum_obliquity(max_obliquity)
    if not has_reindexing_ambiguity(
        dataset,
        max_obliquity=validated_maximum_obliquity,
    ):
        msg = "dataset has no reindexing ambiguity at the requested max_obliquity"
        raise PhaseAlignmentInputError(msg)
    (
        validated_warning_correlation,
        validated_minimum_correlation,
        validated_minimum_correlation_gap,
    ) = _validate_scoring_thresholds(
        warning_correlation=warning_correlation,
        minimum_correlation=minimum_correlation,
        minimum_correlation_gap=minimum_correlation_gap,
    )
    operations = (
        gemmi.Op(IDENTITY_OPERATION),
        *gemmi.find_twin_laws(
            dataset.cell,
            dataset.spacegroup,
            max_obliq=validated_maximum_obliquity,
            all_ops=False,
        ),
    )
    result = _score_reindexing_candidates(
        dataset,
        reference,
        operations,
        data_key=data_key,
        reference_key=reference_key,
        minimum_correlation=validated_minimum_correlation,
        minimum_correlation_gap=validated_minimum_correlation_gap,
    )
    if result.correlation < validated_warning_correlation:
        msg = (
            "selected reindexing operation has low correlation "
            f"{result.correlation:.3f}"
        )
        warn(msg, LowCorrelationWarning, stacklevel=2)
    return result
