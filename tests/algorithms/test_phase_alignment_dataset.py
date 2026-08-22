from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Final, Literal, Optional, cast, get_type_hints

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs

PHASE_DATA_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "data" / "fmodel"
FULL_ROTATION_DEGREES: Final[float] = 360.0
PERMISSIVE_CORRELATION: Final[float] = -1.0
PERMISSIVE_CORRELATION_GAP: Final[float] = 0.0
STRICT_CORRELATION: Final[float] = 0.99
EXPECTED_P61_ORIGIN_SHIFT: Final[tuple[float, float, float]] = (0.0, 0.0, -0.137)
DISTRACTING_P61_ORIGIN_SHIFT: Final[tuple[float, float, float]] = (0.0, 0.0, 0.271)

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]
BooleanArray = NDArray[np.bool_]


@pytest.mark.parametrize(
    "public_object",
    [
        rs.algorithms.has_reindexing_ambiguity,
        rs.algorithms.reindex_by_correlation,
        rs.algorithms.align_phases,
        rs.algorithms.ReindexingResult,
        rs.algorithms.PhaseAlignmentResult,
    ],
)
def test_public_alignment_annotations_resolve(
    public_object: Callable[..., object],
) -> None:
    # Regression: importing DataSet during package initialization must not shadow
    # the concat function.
    dataset_module = importlib.import_module("reciprocalspaceship.dataset")

    assert dataset_module.concat is rs.concat
    assert rs.DataSet in get_type_hints(public_object).values()


def test_align_phases_skips_reindexing_when_no_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miller_indices = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
        dtype=np.int64,
    )
    amplitudes = np.ones(len(miller_indices), dtype=np.float64)
    phases = np.asarray((10.0, -20.0, 70.0, 130.0), dtype=np.float64)
    reference = _synthetic_p1_dataset(miller_indices, amplitudes, phases)
    expected_origin_shift = np.asarray((0.137, -0.271, 0.419), dtype=np.float64)
    moving = _phase_shifted_dataset(
        reference,
        expected_origin_shift,
        phase_key="PHI",
    )
    monkeypatch.setattr(
        "reciprocalspaceship.algorithms.phase_alignment.MAXIMUM_FFT_MEMORY_BYTES",
        800,
    )

    # Regression: skip identity CC scoring and use bounded memory for full-polar data.
    result = _align(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
    )

    assert result.reindexing is None
    np.testing.assert_allclose(result.origin_shift, expected_origin_shift, atol=1e-6)


def _phase_shifted_dataset(
    reference: rs.DataSet,
    origin_shift: FloatArray,
    *,
    phase_key: str = "PHIFMODEL",
) -> rs.DataSet:
    moving = reference.copy()
    phase_shifts = FULL_ROTATION_DEGREES * moving.get_hkls() @ origin_shift
    moving[phase_key] = rs.utils.canonicalize_phases(
        moving[phase_key].to_numpy(dtype=np.float64) + phase_shifts,
    ).astype(np.float32)
    moving[phase_key] = moving[phase_key].astype("Phase")
    return moving


def _synthetic_p1_dataset(
    miller_indices: IntegerArray,
    amplitudes: FloatArray,
    phases: FloatArray,
) -> rs.DataSet:
    dataset = rs.DataSet(
        {
            "H": miller_indices[:, 0],
            "K": miller_indices[:, 1],
            "L": miller_indices[:, 2],
            "F": amplitudes,
            "PHI": phases,
        },
        spacegroup=gemmi.SpaceGroup("P 1"),
        cell=gemmi.UnitCell(41.0, 53.0, 67.0, 79.0, 83.0, 74.0),
        merged=True,
    )
    dataset[["H", "K", "L"]] = dataset[["H", "K", "L"]].astype("HKL")
    dataset["F"] = dataset["F"].astype("SFAmplitude")
    dataset["PHI"] = dataset["PHI"].astype("Phase")
    return dataset.set_index(["H", "K", "L"])


def _p61_datasets_with_conflicting_origin_shifts() -> tuple[
    rs.DataSet, rs.DataSet, BooleanArray
]:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    miller_indices = moving.get_hkls()
    reference_phases = reference["PHIFMODEL"].to_numpy(dtype=np.float64)
    reliable_reflections = np.arange(len(moving)) % 3 == 0
    moving_phases = rs.utils.canonicalize_phases(
        reference_phases
        + FULL_ROTATION_DEGREES
        * miller_indices
        @ np.asarray(DISTRACTING_P61_ORIGIN_SHIFT),
    )
    moving_phases[reliable_reflections] = rs.utils.canonicalize_phases(
        reference_phases[reliable_reflections]
        + FULL_ROTATION_DEGREES
        * miller_indices[reliable_reflections]
        @ np.asarray(EXPECTED_P61_ORIGIN_SHIFT),
    )
    moving["PHIFMODEL"] = moving_phases.astype(np.float32)
    moving["PHIFMODEL"] = moving["PHIFMODEL"].astype("Phase")
    return moving, reference, reliable_reflections


def _align(
    moving: rs.DataSet,
    reference: rs.DataSet,
    *,
    phase_key: str = "PHIFMODEL",
    reference_phase_key: str = "PHIFMODEL",
    amplitude_key: str = "FMODEL",
    reference_amplitude_key: str = "FMODEL",
    weighting: Literal["amplitude", "uniform"] = "amplitude",
    search_hand: bool = False,
    maximum_refinement_starts: int = 4_096,
    fom_key: Optional[str] = None,
    reference_fom_key: Optional[str] = None,
    warning_correlation: float = PERMISSIVE_CORRELATION,
    minimum_correlation: float = PERMISSIVE_CORRELATION,
    minimum_correlation_gap: float = PERMISSIVE_CORRELATION_GAP,
) -> rs.algorithms.PhaseAlignmentResult:
    return rs.algorithms.align_phases(
        moving,
        reference,
        phase_key=phase_key,
        reference_phase_key=reference_phase_key,
        amplitude_key=amplitude_key,
        reference_amplitude_key=reference_amplitude_key,
        weighting=weighting,
        search_hand=search_hand,
        maximum_refinement_starts=maximum_refinement_starts,
        fom_key=fom_key,
        reference_fom_key=reference_fom_key,
        warning_correlation=warning_correlation,
        minimum_correlation=minimum_correlation,
        minimum_correlation_gap=minimum_correlation_gap,
    )


@pytest.mark.parametrize("invalid_value", [0, 1, -1, True, 1.5])
def test_align_phases_rejects_invalid_maximum_refinement_starts(
    invalid_value: object,
) -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))

    # Regression: one refinement start can hide a tied runner-up from the gap gate.
    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="maximum_refinement_starts",
    ):
        _align(
            reference.copy(),
            reference,
            maximum_refinement_starts=cast(int, invalid_value),
        )


def test_align_phases_dataset_smoke_and_phenix_sign() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    expected_origin_shift = np.asarray((0.0, 0.0, -0.137), dtype=np.float64)
    moving = _phase_shifted_dataset(reference, expected_origin_shift)
    original_phases = moving["PHIFMODEL"].copy()

    result = _align(moving, reference)

    assert isinstance(result, rs.algorithms.PhaseAlignmentResult)
    assert isinstance(result.dataset, rs.DataSet)
    assert result.inverted_hand is False
    assert result.reindexing is not None
    assert result.reindexing.operation.triplet() == "x,y,z"
    np.testing.assert_allclose(result.origin_shift, expected_origin_shift, atol=1e-6)
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)
    np.testing.assert_allclose(
        rs.utils.canonicalize_phases(
            result.dataset["PHIFMODEL"].to_numpy(dtype=np.float64)
            - reference["PHIFMODEL"].to_numpy(dtype=np.float64),
        ),
        0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(moving["PHIFMODEL"], original_phases, atol=1e-12)


def test_align_phases_matches_common_hkls_independent_of_row_order() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    expected_origin_shift = np.asarray(EXPECTED_P61_ORIGIN_SHIFT, dtype=np.float64)
    moving = _phase_shifted_dataset(reference, expected_origin_shift)
    moving = moving.iloc[7:].sample(frac=1.0, random_state=20260822)
    reference = reference.iloc[:-11].sample(frac=1.0, random_state=20260823)

    # Regression: the DataSet interface must align by HKL rather than row position.
    result = _align(moving, reference)

    np.testing.assert_allclose(result.origin_shift, expected_origin_shift, atol=1e-6)
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)


def test_align_phases_reindexes_before_origin_search() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    reindexing_operation = reference.reindexing_ops[0]
    moving = reference.apply_symop(reindexing_operation)

    # Regression: alternate P61 indexing must be corrected before fitting the origin.
    result = _align(moving, reference)

    assert result.reindexing is not None
    assert result.reindexing.operation.triplet() == reindexing_operation.triplet()
    np.testing.assert_allclose(result.reindexing.correlation, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.origin_shift, 0.0, atol=1e-6)


def test_align_phases_constrained_fft_recovers_noisy_p61_regression() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    expected_origin_shift = np.asarray((0.0, 0.0, -0.137), dtype=np.float64)
    moving = _phase_shifted_dataset(reference, expected_origin_shift)
    random_number_generator = np.random.default_rng(seed=20260904)
    random_number_generator.normal(loc=0.0, scale=90.0, size=len(moving))
    noise = random_number_generator.normal(loc=0.0, scale=90.0, size=len(moving))
    moving["PHIFMODEL"] += noise

    # Regression: the former shared 3-D FFT seed converged to z=0.469 for this trial.
    result = _align(moving, reference)

    np.testing.assert_allclose(
        result.origin_shift,
        expected_origin_shift,
        atol=1e-3,
    )


def test_align_phases_uses_amplitude_product_weights_by_default() -> None:
    random_number_generator = np.random.default_rng(seed=20260814)
    grid = np.mgrid[-7:8, -7:8, -7:8].reshape(3, -1).T
    grid = grid[np.any(grid != 0, axis=1)]
    grid = grid[rs.utils.in_asu(grid, gemmi.SpaceGroup("P 1"))]
    random_number_generator.shuffle(grid)
    number_of_reliable_reflections = 500
    number_of_unreliable_reflections = 800
    miller_indices = np.asarray(
        grid[: number_of_reliable_reflections + number_of_unreliable_reflections],
        dtype=np.int64,
    )
    reference_phases = random_number_generator.uniform(
        low=-180.0,
        high=180.0,
        size=len(miller_indices),
    )
    expected_origin_shift = np.asarray((0.173, -0.281, 0.419), dtype=np.float64)
    distracting_origin_shift = np.asarray((-0.375, 0.125, 0.25), dtype=np.float64)
    moving_phases = rs.utils.canonicalize_phases(
        reference_phases
        + FULL_ROTATION_DEGREES * miller_indices @ distracting_origin_shift,
    )
    moving_phases[:number_of_reliable_reflections] = rs.utils.canonicalize_phases(
        reference_phases[:number_of_reliable_reflections]
        + FULL_ROTATION_DEGREES
        * miller_indices[:number_of_reliable_reflections]
        @ expected_origin_shift,
    )
    amplitudes = np.full(len(miller_indices), 0.01, dtype=np.float64)
    amplitudes[:number_of_reliable_reflections] = 10.0
    reference = _synthetic_p1_dataset(
        miller_indices,
        amplitudes,
        np.asarray(reference_phases, dtype=np.float64),
    )
    moving = _synthetic_p1_dataset(
        miller_indices,
        amplitudes,
        np.asarray(moving_phases, dtype=np.float64),
    )

    weighted_result = _align(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
    )
    uniform_result = _align(
        moving,
        reference,
        phase_key="PHI",
        reference_phase_key="PHI",
        amplitude_key="F",
        reference_amplitude_key="F",
        weighting="uniform",
    )

    np.testing.assert_allclose(
        weighted_result.origin_shift,
        expected_origin_shift,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        uniform_result.origin_shift,
        distracting_origin_shift,
        atol=1e-3,
    )


def test_align_phases_multiplies_fom_into_phase_weights() -> None:
    moving, reference, reliable_reflections = (
        _p61_datasets_with_conflicting_origin_shifts()
    )
    unweighted_result = _align(moving, reference)
    figures_of_merit = np.zeros(len(moving), dtype=np.float32)
    figures_of_merit[reliable_reflections] = 1.0
    for dataset in (moving, reference):
        dataset["FOM"] = figures_of_merit
        dataset["FOM"] = dataset["FOM"].astype("Weight")

    weighted_result = _align(
        moving,
        reference,
        fom_key="FOM",
        reference_fom_key="FOM",
    )

    np.testing.assert_allclose(
        unweighted_result.origin_shift,
        DISTRACTING_P61_ORIGIN_SHIFT,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        weighted_result.origin_shift,
        EXPECTED_P61_ORIGIN_SHIFT,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("target", "invalid_figure_of_merit"),
    [("moving", -0.01), ("reference", 1.01)],
)
def test_align_phases_rejects_out_of_range_fom_values(
    target: str,
    invalid_figure_of_merit: float,
) -> None:
    moving, reference, _ = _p61_datasets_with_conflicting_origin_shifts()
    moving_fom = np.ones(len(moving), dtype=np.float32)
    reference_fom = np.ones(len(reference), dtype=np.float32)
    if target == "moving":
        moving_fom[0] = invalid_figure_of_merit
    else:
        reference_fom[0] = invalid_figure_of_merit
    moving["FOM"] = moving_fom
    reference["FOM"] = reference_fom
    moving["FOM"] = moving["FOM"].astype("Weight")
    reference["FOM"] = reference["FOM"].astype("Weight")

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="zero and one"):
        _align(
            moving,
            reference,
            fom_key="FOM",
            reference_fom_key="FOM",
        )


def test_align_phases_rejects_nonnumeric_fom_values() -> None:
    moving, reference, _ = _p61_datasets_with_conflicting_origin_shifts()
    moving["FOM"] = ["not-a-number"] * len(moving)
    reference["FOM"] = np.ones(len(reference), dtype=np.float32)

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="FOM"):
        _align(
            moving,
            reference,
            fom_key="FOM",
            reference_fom_key="FOM",
        )


def test_align_phases_requires_named_fom_columns() -> None:
    moving, reference, _ = _p61_datasets_with_conflicting_origin_shifts()

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="FOM key"):
        _align(
            moving,
            reference,
            fom_key="FOM",
            reference_fom_key="FOM",
        )


def test_align_phases_rejects_zero_fom_weights() -> None:
    moving, reference, _ = _p61_datasets_with_conflicting_origin_shifts()
    moving["FOM"] = np.zeros(len(moving), dtype=np.float32)
    reference["FOM"] = np.zeros(len(reference), dtype=np.float32)
    moving["FOM"] = moving["FOM"].astype("Weight")
    reference["FOM"] = reference["FOM"].astype("Weight")

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="positive"):
        _align(
            moving,
            reference,
            fom_key="FOM",
            reference_fom_key="FOM",
        )


def test_align_phases_requires_three_finite_phase_reflections() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    moving_phases = np.full(len(moving), np.nan, dtype=np.float32)
    moving_phases[:2] = moving["PHIFMODEL"].to_numpy(dtype=np.float32)[:2]
    moving["PHIFMODEL"] = moving_phases
    moving["PHIFMODEL"] = moving["PHIFMODEL"].astype("Phase")

    with pytest.raises(
        rs.algorithms.PhaseAlignmentInputError,
        match="at least 3 finite phase reflections",
    ):
        _align(moving, reference)


def test_align_phases_intensity_weights_match_amplitude_weights() -> None:
    moving, reference, _ = _p61_datasets_with_conflicting_origin_shifts()
    for dataset in (moving, reference):
        dataset["I"] = dataset["FMODEL"].to_numpy(dtype=np.float64) ** 2
        dataset["I"] = dataset["I"].astype("Intensity")

    amplitude_result = _align(moving, reference)
    intensity_result = _align(
        moving,
        reference,
        amplitude_key="I",
        reference_amplitude_key="I",
    )

    np.testing.assert_allclose(
        intensity_result.origin_shift,
        amplitude_result.origin_shift,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [candidate.correlation for candidate in intensity_result.candidates],
        [candidate.correlation for candidate in amplitude_result.candidates],
        atol=1e-7,
    )


def test_align_phases_searches_hand_only_when_requested() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "3KXE.mtz"))
    moving = reference.copy()
    moving["PHIFMODEL"] = -moving["PHIFMODEL"]

    # Regression: hand inversion is distinct from a proper reindexing operation.
    with pytest.raises(rs.algorithms.NoClearSolutionError):
        _align(
            moving,
            reference,
            minimum_correlation=STRICT_CORRELATION,
        )

    result = _align(
        moving,
        reference,
        search_hand=True,
        minimum_correlation=STRICT_CORRELATION,
    )

    assert result.inverted_hand is True
    np.testing.assert_allclose(result.correlation, 1.0, atol=1e-10)


def test_align_phases_transforms_complex_columns_for_hand_and_origin() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "3KXE.mtz"))
    expected_origin_shift = np.asarray((-0.5, 0.0, -0.5), dtype=np.float64)
    miller_indices = reference.get_hkls()
    amplitudes = reference["FMODEL"].to_numpy(dtype=np.float64)
    reference_phases = reference["PHIFMODEL"].to_numpy(dtype=np.float64)
    reference["FCOMPLEX"] = amplitudes * np.exp(1j * np.deg2rad(reference_phases))
    moving = reference.copy()
    moving_phases = rs.utils.canonicalize_phases(
        -reference_phases
        - FULL_ROTATION_DEGREES * miller_indices @ expected_origin_shift,
    )
    moving["PHIFMODEL"] = moving_phases.astype(np.float32)
    moving["PHIFMODEL"] = moving["PHIFMODEL"].astype("Phase")
    moving["FCOMPLEX"] = amplitudes * np.exp(1j * np.deg2rad(moving_phases))
    original_complex_values = moving["FCOMPLEX"].to_numpy(copy=True)

    result = _align(
        moving,
        reference,
        search_hand=True,
        minimum_correlation=STRICT_CORRELATION,
    )

    assert result.inverted_hand is True
    np.testing.assert_allclose(
        result.origin_shift,
        expected_origin_shift,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.dataset["FCOMPLEX"],
        reference["FCOMPLEX"],
        atol=1e-10,
    )
    np.testing.assert_allclose(moving["FCOMPLEX"], original_complex_values, atol=0.0)


def test_align_phases_rejects_nonisomorphous_inputs() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    moving.cell = gemmi.UnitCell(
        2.0 * reference.cell.a,
        reference.cell.b,
        reference.cell.c,
        reference.cell.alpha,
        reference.cell.beta,
        reference.cell.gamma,
    )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="isomorphous"):
        _align(moving, reference)


def test_align_phases_warns_for_low_but_usable_correlation() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    random_number_generator = np.random.default_rng(seed=20260814)
    moving["PHIFMODEL"] += random_number_generator.normal(
        loc=0.0,
        scale=60.0,
        size=len(moving),
    )

    with pytest.warns(rs.algorithms.LowCorrelationWarning):
        result = _align(
            moving,
            reference,
            warning_correlation=0.99,
        )

    assert result.correlation < 0.99


def test_align_phases_warns_for_low_reindexing_correlation() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.apply_symop(reference.reindexing_ops[0])
    random_number_generator = np.random.default_rng(seed=20260822)
    moving["FMODEL"] *= random_number_generator.lognormal(
        mean=0.0,
        sigma=0.1,
        size=len(moving),
    )
    moving["FMODEL"] = moving["FMODEL"].astype("SFAmplitude")

    with pytest.warns(
        rs.algorithms.LowCorrelationWarning,
        match="reindexing",
    ) as warning_records:
        result = rs.algorithms.align_phases(
            moving,
            reference,
            phase_key="PHIFMODEL",
            reference_phase_key="PHIFMODEL",
            amplitude_key="FMODEL",
            reference_amplitude_key="FMODEL",
            warning_correlation=PERMISSIVE_CORRELATION,
            minimum_correlation=PERMISSIVE_CORRELATION,
            minimum_correlation_gap=PERMISSIVE_CORRELATION_GAP,
            reindexing_warning_correlation=1.0,
            reindexing_minimum_correlation=PERMISSIVE_CORRELATION,
            reindexing_minimum_correlation_gap=PERMISSIVE_CORRELATION_GAP,
        )

    assert result.reindexing is not None
    assert result.reindexing.correlation < 1.0
    assert warning_records[0].filename == __file__


def test_align_phases_raises_when_no_clear_solution_is_found() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "6OVT.mtz"))
    moving = reference.copy()
    random_number_generator = np.random.default_rng(seed=20260814)
    moving["PHIFMODEL"] = random_number_generator.uniform(
        low=-180.0,
        high=180.0,
        size=len(moving),
    ).astype(np.float32)
    moving["PHIFMODEL"] = moving["PHIFMODEL"].astype("Phase")

    with pytest.raises(rs.algorithms.NoClearSolutionError, match="correlation"):
        _align(
            moving,
            reference,
            minimum_correlation=0.9,
        )


def test_align_phases_raises_when_origin_candidates_are_tied() -> None:
    reference = rs.read_mtz(str(PHASE_DATA_DIRECTORY / "3KXE.mtz"))
    even_indices = np.all(reference.get_hkls() % 2 == 0, axis=1)
    reference = reference.iloc[np.flatnonzero(even_indices)].copy()

    # Regression: all-even HKLs cannot distinguish the half-cell origin choices.
    with pytest.raises(rs.algorithms.NoClearSolutionError, match="correlation gap"):
        _align(
            reference.copy(),
            reference,
            minimum_correlation_gap=0.01,
        )
