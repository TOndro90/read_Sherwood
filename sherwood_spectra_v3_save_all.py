#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_sherwood_file(filename):
    with open(filename, "rb") as f:
        npix = np.fromfile(f, dtype=np.int32, count=1)[0]
        nlos = np.fromfile(f, dtype=np.int32, count=1)[0]

        ztime = np.fromfile(f, dtype=np.float32, count=1)[0]
        omegaM = np.fromfile(f, dtype=np.float32, count=1)[0]
        omegaL = np.fromfile(f, dtype=np.float32, count=1)[0]
        omegab = np.fromfile(f, dtype=np.float32, count=1)[0]
        hubble = np.fromfile(f, dtype=np.float32, count=1)[0]
        boxsize = np.fromfile(f, dtype=np.float32, count=1)[0]

        axis = np.fromfile(f, dtype=np.int32, count=nlos)
        coord1 = np.fromfile(f, dtype=np.float32, count=nlos)
        coord2 = np.fromfile(f, dtype=np.float32, count=nlos)

        pixels = np.fromfile(f, dtype=np.float32, count=npix)

        tau = np.fromfile(f, dtype=np.float32, count=npix * nlos)
        tau = tau.reshape((nlos, npix))

    return {
        "npix": npix,
        "nlos": nlos,
        "ztime": ztime,
        "omegaM": omegaM,
        "omegaL": omegaL,
        "omegab": omegab,
        "hubble": hubble,
        "boxsize": boxsize,
        "axis": axis,
        "coord1": coord1,
        "coord2": coord2,
        "pixels": pixels,
        "tau": tau,
    }


def wavelength_from_pixels(pixels, ztime, boxsize):
    lambda_alpha = 1215.67  # Angstrom
    return lambda_alpha * (1.0 + ztime) * (1.0 + pixels / boxsize)


def save_one_los(data, los, outfile):
    wavelength = wavelength_from_pixels(
        data["pixels"], data["ztime"], data["boxsize"]
    )
    tau_los = data["tau"][los]

    arr = np.column_stack([wavelength, tau_los])

    header = (
        "wavelength_A tau\n"
        f"redshift={data['ztime']:.5f}\n"
        f"los={los}\n"
        f"axis={data['axis'][los]}\n"
        f"coord1={data['coord1'][los]:.6f}\n"
        f"coord2={data['coord2'][los]:.6f}"
    )

    np.savetxt(outfile, arr, header=header)


def save_all_los(data, input_filename, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Path(input_filename).stem

    for los in range(data["nlos"]):
        outfile = output_dir / f"{base}_los{los:05d}.txt"
        save_one_los(data, los, outfile)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Read Sherwood simulation spectra"
    )

    parser.add_argument(
        "filename",
        help="Sherwood binary spectra file"
    )

    parser.add_argument(
        "plotlos",
        nargs="?",
        type=int,
        default=0,
        help="LOS index"
    )

    parser.add_argument(
        "--treecool",
        type=str,
        default=None,
        help="TREECOOL UV background file"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display spectrum plot for selected LOS"
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save selected LOS wavelength vs tau to text file"
    )

    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save every LOS into separate text files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text filename for --save-txt"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="los_txt",
        help="Output directory for --save-all"
    )

    args = parser.parse_args()

    data = read_sherwood_file(args.filename)

    los = args.plotlos

    if los < 0 or los >= data["nlos"]:
        raise ValueError(f"LOS index must be between 0 and {data['nlos'] - 1}")

    tau_los = data["tau"][los]
    wavelength = wavelength_from_pixels(
        data["pixels"], data["ztime"], data["boxsize"]
    )

    print(f"Redshift: {data['ztime']:.3f}")
    print(f"Number of LOS: {data['nlos']}")
    print(f"Pixels per LOS: {data['npix']}")
    print(f"Selected LOS: {los}")
    print(f"Mean tau selected LOS: {tau_los.mean():.5f}")

    if args.save_txt:
        if args.output is None:
            base = Path(args.filename).stem
            outfile = f"{base}_los{los:05d}.txt"
        else:
            outfile = args.output

        save_one_los(data, los, outfile)
        print(f"Saved selected LOS text file: {outfile}")

    if args.save_all:
        outdir = save_all_los(data, args.filename, args.output_dir)
        print(f"Saved all LOS files to directory: {outdir}")

    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.plot(wavelength, tau_los)
        plt.xlabel("Wavelength [Angstrom]")
        plt.ylabel("Optical depth tau")
        plt.title(f"Sherwood LOS {los} z={data['ztime']:.2f}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
