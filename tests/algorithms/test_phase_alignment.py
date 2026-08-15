from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray

import reciprocalspaceship as rs


PHASE_DATA_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "data" / "fmodel"
FULL_ROTATION_DEGREES: Final[float] = 360.0

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


@dataclass(frozen=True)
class PhaseAlignmentCase:
    """Inputs and expected translation for a phase-alignment test."""

    miller_indices: IntegerArray
    phases: FloatArray
    reference_phases: FloatArray
    spacegroup: gemmi.SpaceGroup
    expected_translation: FloatArray


def _translated_phases(
    miller_indices: IntegerArray,
    phases: FloatArray,
    fractional_translation: FloatArray,
) -> FloatArray:
    phase_shifts = FULL_ROTATION_DEGREES * miller_indices @ fractional_translation
    return np.asarray(rs.utils.canonicalize_phases(phases + phase_shifts), dtype=np.float64)


def _load_phase_alignment_case(
    filename: str,
    expected_translation: tuple[float, float, float],
) -> PhaseAlignmentCase:
    dataset = rs.read_mtz(str(PHASE_DATA_DIRECTORY / filename))
    miller_indices = np.asarray(dataset.get_hkls(), dtype=np.int64)
    reference_phases = dataset["PHIFMODEL"].to_numpy(dtype=np.float64)
    translation = np.asarray(expected_translation, dtype=np.float64)
    phases = _translated_phases(miller_indices, reference_phases, -translation)
    return PhaseAlignmentCase(
        miller_indices=miller_indices,
        phases=phases,
        reference_phases=reference_phases,
        spacegroup=dataset.spacegroup,
        expected_translation=translation,
    )


@pytest.fixture(scope="session")
def p43212_case() -> PhaseAlignmentCase:
    return _load_phase_alignment_case("9LYZ.mtz", (0.5, 0.5, 0.5))


@pytest.fixture(scope="session")
def p212121_case() -> PhaseAlignmentCase:
    return _load_phase_alignment_case("3KXE.mtz", (0.5, 0.0, 0.5))


@pytest.fixture(scope="session")
def p61_polar_case() -> PhaseAlignmentCase:
    return _load_phase_alignment_case("6OVT.mtz", (0.0, 0.0, 0.137))


@pytest.fixture(scope="session")
def r3r_polar_case() -> PhaseAlignmentCase:
    random_number_generator = np.random.default_rng(seed=20260814)
    miller_indices = random_number_generator.integers(low=-8, high=9, size=(2_000, 3))
    reference_phases = random_number_generator.uniform(low=-180.0, high=180.0, size=2_000)
    expected_translation = np.asarray((0.137, 0.137, 0.137), dtype=np.float64)
    phases = _translated_phases(
        miller_indices,
        reference_phases,
        -expected_translation,
    )
    return PhaseAlignmentCase(
        miller_indices=miller_indices,
        phases=phases,
        reference_phases=reference_phases,
        spacegroup=gemmi.SpaceGroup("R 3:R"),
        expected_translation=expected_translation,
    )


@pytest.mark.parametrize(
    "case_fixture_name",
    ["p43212_case", "p212121_case", "p61_polar_case", "r3r_polar_case"],
)
def test_align_phases_smoke(
    case_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    case: PhaseAlignmentCase = request.getfixturevalue(case_fixture_name)

    aligned_phases, fractional_translation = rs.algorithms.align_phases(
        case.miller_indices,
        case.phases,
        case.reference_phases,
        case.spacegroup,
    )

    assert isinstance(aligned_phases, np.ndarray)
    assert isinstance(fractional_translation, np.ndarray)
    assert aligned_phases.dtype == np.float64
    assert fractional_translation.dtype == np.float64
    assert aligned_phases.shape == case.phases.shape
    assert fractional_translation.shape == (3,)
    assert np.logical_and(aligned_phases >= -180.0, aligned_phases < 180.0).all()
    np.testing.assert_allclose(fractional_translation, case.expected_translation, atol=1e-6)
    phase_residuals = rs.utils.canonicalize_phases(
        aligned_phases - case.reference_phases
    )
    np.testing.assert_allclose(phase_residuals, 0.0, atol=1e-5)


def test_align_phases_snaps_nonpolar_origin_with_noise(
    p43212_case: PhaseAlignmentCase,
) -> None:
    # Regression: noise must not move a nonpolar fit away from an allowed origin.
    random_number_generator = np.random.default_rng(seed=20260814)
    noisy_phases = rs.utils.canonicalize_phases(
        p43212_case.phases
        + random_number_generator.normal(
            loc=0.0,
            scale=10.0,
            size=len(p43212_case.phases),
        )
    )

    _, fractional_translation = rs.algorithms.align_phases(
        p43212_case.miller_indices,
        noisy_phases,
        p43212_case.reference_phases,
        p43212_case.spacegroup,
    )

    np.testing.assert_allclose(
        fractional_translation,
        p43212_case.expected_translation,
        atol=1e-12,
    )


def test_align_phases_uses_weights_for_p1() -> None:
    random_number_generator = np.random.default_rng(seed=20260814)
    number_of_reliable_reflections = 1_200
    number_of_unreliable_reflections = 1_800
    number_of_reflections = (
        number_of_reliable_reflections + number_of_unreliable_reflections
    )
    miller_indices = random_number_generator.integers(
        low=-12,
        high=13,
        size=(number_of_reflections, 3),
    )
    reference_phases = random_number_generator.uniform(
        low=-180.0,
        high=180.0,
        size=number_of_reflections,
    )
    expected_translation = np.asarray((0.173, 0.419, 0.731), dtype=np.float64)
    incorrect_translation = np.asarray((0.625, 0.125, 0.375), dtype=np.float64)
    phases = _translated_phases(
        miller_indices,
        reference_phases,
        -incorrect_translation,
    )
    phases[:number_of_reliable_reflections] = _translated_phases(
        miller_indices[:number_of_reliable_reflections],
        reference_phases[:number_of_reliable_reflections],
        -expected_translation,
    )
    phases[:number_of_reliable_reflections] += random_number_generator.normal(
        loc=0.0,
        scale=5.0,
        size=number_of_reliable_reflections,
    )
    weights = np.zeros(number_of_reflections, dtype=np.float64)
    weights[:number_of_reliable_reflections] = 1.0

    _, fractional_translation = rs.algorithms.align_phases(
        miller_indices,
        phases,
        reference_phases,
        gemmi.SpaceGroup("P 1"),
        weights=weights,
    )

    np.testing.assert_allclose(
        fractional_translation,
        expected_translation,
        atol=2e-4,
    )


def test_align_phases_rejects_invalid_inputs() -> None:
    miller_indices = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    phases = np.asarray((10.0, 20.0, 30.0))
    spacegroup = gemmi.SpaceGroup("P 1")

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="shape"):
        rs.algorithms.align_phases(
            miller_indices[:, :2],
            phases,
            phases,
            spacegroup,
        )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="same length"):
        rs.algorithms.align_phases(
            miller_indices,
            phases[:-1],
            phases,
            spacegroup,
        )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="finite"):
        rs.algorithms.align_phases(
            miller_indices,
            np.asarray((10.0, np.nan, 30.0)),
            phases,
            spacegroup,
        )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="nonnegative"):
        rs.algorithms.align_phases(
            miller_indices,
            phases,
            phases,
            spacegroup,
            weights=np.asarray((1.0, -1.0, 1.0)),
        )

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="positive"):
        rs.algorithms.align_phases(
            miller_indices,
            phases,
            phases,
            spacegroup,
            weights=np.zeros(3),
        )


def test_align_phases_rejects_unidentifiable_p1_translation() -> None:
    miller_indices = np.asarray(((1, 0, 0), (2, 0, 0), (3, 0, 0)))
    phases = np.asarray((10.0, 20.0, 30.0))

    with pytest.raises(rs.algorithms.PhaseAlignmentInputError, match="identify"):
        rs.algorithms.align_phases(
            miller_indices,
            phases,
            phases,
            gemmi.SpaceGroup("P 1"),
        )
