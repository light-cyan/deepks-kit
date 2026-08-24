"""Method-neutral restricted orbital-response contractions."""

import numpy as np


def density_from_mo_response(
    mo_response: np.ndarray,
    coefficient: np.ndarray,
    occupation: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:
    """Build a restricted AO density response from one MO response."""
    occupied_coefficients = coefficient[:, occupied]
    coefficient_response = np.einsum(
        "mp,...pi->...mi",
        coefficient,
        mo_response,
    )
    one_sided = np.einsum(
        "...pi,qi,i->...pq",
        coefficient_response,
        occupied_coefficients,
        occupation[occupied],
    )
    return one_sided + one_sided.swapaxes(-1, -2)


class RestrictedResponseAlgebra:
    """Share exact RHF/RKS density and induced-potential contractions."""

    def _density_from_mo_response(
        self,
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        return density_from_mo_response(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )

    def _mo_potential(
        self,
        potential: np.ndarray,
        coefficient: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        return np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            potential,
            coefficient[:, occupied],
        )

    def _induced_mo_potential(
        self,
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        density_response = self._density_from_mo_response(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        return self._mo_potential(
            self._induced_potential(density_response),
            coefficient,
            occupied,
        )


__all__ = ["RestrictedResponseAlgebra", "density_from_mo_response"]
