"""Python reader for Sherwood mock Lyman-alpha forest spectra.

Converted from the public IDL routines:
- read_spectra.pro
- get_los_position.pro
- get_uvb.pro
- get_wavelength.pro

The binary layout follows the Sherwood public data README: npix, nlos,
cosmology fields, LOS arrays, pixel positions, and tau values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LYA_REST_ANGSTROM = 1215.6701
C_KMS = 2.9979e5


@dataclass
class SherwoodSpectra:
    npix: int
    nlos: int
    ztime: float
    omega_m: float
    omega_l: float
    omega_b: float
    hubble: float
    boxsize: float
    axis: np.ndarray      # shape (nlos,), x=1, y=2, z=3
    coord1: np.ndarray    # shape (nlos,), h^-1 ckpc
    coord2: np.ndarray    # shape (nlos,), h^-1 ckpc
    pixels: np.ndarray    # shape (npix,), h^-1 ckpc
    tau: np.ndarray       # shape (nlos, npix)
    wavelength: np.ndarray  # shape (npix,), observed-frame Angstroms
    gamma_hi: Optional[float] = None  # s^-1, from TREECOOL table if supplied

    @property
    def flux(self) -> np.ndarray:
        """Transmitted flux, F = exp(-tau), shape (nlos, npix)."""
        return np.exp(-self.tau)

    @property
    def tau_eff(self) -> float:
        """Effective optical depth averaged over all pixels and sight-lines."""
        return float(-np.log(np.mean(np.exp(-self.tau))))

    def los_position(self, plotlos: int) -> Dict[str, float | int | str]:
        """Return the position metadata for one line of sight."""
        _check_los(plotlos, self.nlos)
        ax = int(self.axis[plotlos])
        c1 = float(self.coord1[plotlos])
        c2 = float(self.coord2[plotlos])

        if ax == 1:
            return {"los": plotlos, "axis": ax, "drawn_along": "x", "y_hinv_ckpc": c1, "z_hinv_ckpc": c2}
        if ax == 2:
            return {"los": plotlos, "axis": ax, "drawn_along": "y", "x_hinv_ckpc": c1, "z_hinv_ckpc": c2}
        if ax == 3:
            return {"los": plotlos, "axis": ax, "drawn_along": "z", "x_hinv_ckpc": c1, "y_hinv_ckpc": c2}
        raise ValueError(f"Unexpected LOS axis value {ax}; expected 1, 2, or 3.")

    def plot_flux(self, plotlos: int):
        """Plot transmitted flux against observed wavelength for one LOS.

        Requires matplotlib. Returns (fig, ax).
        """
        _check_los(plotlos, self.nlos)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.wavelength, self.flux[plotlos])
        ax.set_xlim(float(np.min(self.wavelength)), float(np.max(self.wavelength)))
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Wavelength [Angstroms]")
        ax.set_ylabel("Transmitted flux")
        return fig, ax


def read_spectra(
    filename: str | Path,
    plotlos: Optional[int] = None,
    treecool_file: str | Path | None = None,
    endian: str = "<",
    make_plot: bool = False,
) -> SherwoodSpectra:
    """Read a Sherwood binary spectra file.

    Parameters
    ----------
    filename:
        Path to a Sherwood ``tauH1_lya_zX.X.dat`` binary file.
    plotlos:
        Optional zero-based line-of-sight index. If given, the index is checked
        and a position summary is printed, matching the original IDL workflow.
    treecool_file:
        Optional path to ``TREECOOL_HM12_G+Q``. If supplied, the HI
        photo-ionisation rate is interpolated to the spectrum redshift.
    endian:
        Byte order for the binary file. ``"<"`` is little-endian and works for
        the public files on typical machines. Use ``">"`` for big-endian files.
    make_plot:
        If True, plot transmitted flux for ``plotlos``. Requires plotlos.

    Returns
    -------
    SherwoodSpectra
        Container with arrays, cosmology, wavelength, tau_eff, flux, and helpers.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Spectra file not found: {filename}")

    i4 = np.dtype(endian + "i4")
    f4 = np.dtype(endian + "f4")

    with filename.open("rb") as fh:
        header_ints = np.fromfile(fh, dtype=i4, count=2)
        if header_ints.size != 2:
            raise ValueError("Could not read npix and nlos from file header.")
        npix, nlos = map(int, header_ints)

        header_floats = np.fromfile(fh, dtype=f4, count=6)
        if header_floats.size != 6:
            raise ValueError("Could not read cosmology/header floats.")
        ztime, omega_m, omega_l, omega_b, hubble, boxsize = map(float, header_floats)

        axis = np.fromfile(fh, dtype=i4, count=nlos)
        coord1 = np.fromfile(fh, dtype=f4, count=nlos)
        coord2 = np.fromfile(fh, dtype=f4, count=nlos)
        pixels = np.fromfile(fh, dtype=f4, count=npix)
        tau_flat = np.fromfile(fh, dtype=f4, count=npix * nlos)

    if axis.size != nlos or coord1.size != nlos or coord2.size != nlos:
        raise ValueError("File ended while reading LOS metadata arrays.")
    if pixels.size != npix:
        raise ValueError("File ended while reading pixel-position array.")
    if tau_flat.size != npix * nlos:
        raise ValueError("File ended while reading tau array.")

    tau = tau_flat.reshape((nlos, npix))
    wavelength = get_wavelength(pixels, ztime, omega_m, omega_l, hubble)

    gamma_hi = None
    if treecool_file is not None:
        gamma_hi = get_uvb(treecool_file, ztime)

    spectra = SherwoodSpectra(
        npix=npix,
        nlos=nlos,
        ztime=ztime,
        omega_m=omega_m,
        omega_l=omega_l,
        omega_b=omega_b,
        hubble=hubble,
        boxsize=boxsize,
        axis=axis,
        coord1=coord1,
        coord2=coord2,
        pixels=pixels,
        tau=tau,
        wavelength=wavelength,
        gamma_hi=gamma_hi,
    )

    print(f"Reading spectra at z={spectra.ztime:g}")
    if plotlos is not None:
        print_los_position(spectra, plotlos)
    print(f"The effective optical depth is {spectra.tau_eff:g}")
    if spectra.gamma_hi is not None:
        print(f"The H1 photo-ionisation rate is {spectra.gamma_hi:g} s^-1")

    if make_plot:
        if plotlos is None:
            raise ValueError("make_plot=True requires plotlos to be set.")
        spectra.plot_flux(plotlos)

    return spectra


def get_los_position(axis: np.ndarray, coord1: np.ndarray, coord2: np.ndarray, plotlos: int) -> Dict[str, float | int | str]:
    """Standalone LOS-position helper matching get_los_position.pro."""
    nlos = len(axis)
    _check_los(plotlos, nlos)
    dummy = SherwoodSpectra(
        npix=0, nlos=nlos, ztime=np.nan, omega_m=np.nan, omega_l=np.nan,
        omega_b=np.nan, hubble=np.nan, boxsize=np.nan, axis=np.asarray(axis),
        coord1=np.asarray(coord1), coord2=np.asarray(coord2), pixels=np.array([]),
        tau=np.empty((nlos, 0)), wavelength=np.array([]), gamma_hi=None,
    )
    return dummy.los_position(plotlos)


def print_los_position(spectra: SherwoodSpectra, plotlos: int) -> None:
    """Print LOS position in the same style as the original IDL routine."""
    pos = spectra.los_position(plotlos)
    print(f"LOS {plotlos} is drawn along the {pos['drawn_along']}-axis")
    for key, value in pos.items():
        if key.endswith("_hinv_ckpc"):
            label = key.replace("_hinv_ckpc", " [h^-1 ckpc]")
            print(f"{label} = {value:g}")


def get_uvb(treecool_file: str | Path, ztime: float) -> float:
    """Interpolate the TREECOOL HM12 G+Q HI photo-ionisation rate to ztime.

    The IDL code reads the first two columns. The first column is log10(1+z),
    so z = 10**column0 - 1; the second column is Gamma_HI in s^-1.
    """
    data = np.loadtxt(treecool_file)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("TREECOOL table must contain at least two columns.")
    redshift = np.power(10.0, data[:, 0]) - 1.0
    gamma_hi = data[:, 1]
    order = np.argsort(redshift)
    return float(np.interp(ztime, redshift[order], gamma_hi[order]))


def get_wavelength(
    pixels: np.ndarray,
    ztime: float,
    omega_m: float,
    omega_l: float,
    hubble: float,
) -> np.ndarray:
    """Convert pixel positions in h^-1 ckpc to observed-frame wavelength."""
    pixels = np.asarray(pixels, dtype=np.float64)
    hz = 100.0 * hubble * np.sqrt(omega_m * (1.0 + ztime) ** 3.0 + omega_l)
    pixels_pmpc = 1.0e-3 * pixels / (hubble * (1.0 + ztime))
    vel = hz * pixels_pmpc
    wvel = vel / C_KMS
    return LYA_REST_ANGSTROM * (1.0 + ztime) * np.sqrt((1.0 + wvel) / (1.0 - wvel))


def _check_los(plotlos: int, nlos: int) -> None:
    if plotlos < 0 or plotlos > nlos - 1:
        raise IndexError(f"plotlos must be between 0 and {nlos - 1}, got {plotlos}.")


if __name__ == "__main__":
    # Example:
    # python sherwood_spectra.py ./spectra_40_2048/tauH1_lya_z2.0.dat 3982 ./TREECOOL_HM12_G+Q
    import argparse

    parser = argparse.ArgumentParser(description="Read Sherwood mock Lyman-alpha spectra.")
    parser.add_argument("filename", help="Path to tauH1_lya_zX.X.dat")
    parser.add_argument("plotlos", nargs="?", type=int, default=None, help="Optional LOS index, 0-based")
    parser.add_argument("--treecool", default=None, help="Path to TREECOOL_HM12_G+Q")
    parser.add_argument("--plot", action="store_true", help="Plot flux for plotlos")
    args = parser.parse_args()

    read_spectra(args.filename, plotlos=args.plotlos, treecool_file=args.treecool, make_plot=args.plot)
