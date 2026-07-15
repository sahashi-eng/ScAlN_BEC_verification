#!/usr/bin/env python3
"""Evaluate a fine-tuned Equivar model against BECs stored in extxyz."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from equivar_eval.scripts.calculate import Calculate


DIAGONAL_INDICES = np.array([0, 4, 8])
DIAGONAL_LABELS = ("xx", "yy", "zz")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Equivar predictions with per-atom Born effective charge "
            "tensors stored as becs:R:9 in an extended XYZ dataset."
        )
    )
    parser.add_argument(
        "--test_xyz",
        type=Path,
        required=True,
        metavar="YOUR_PATH_TO_TEST_XYZ",
        help="Extended XYZ test dataset containing a per-atom becs:R:9 array.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        metavar="YOUR_PATH_TO_FINETUNED_MODEL",
        help="Fine-tuned TorchScript Equivar model.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("bec_evaluation"),
        metavar="YOUR_PATH_TO_OUTPUT_DIR",
        help="Output directory (default: ./bec_evaluation).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument("--graph_max_radius", type=float, default=3.0)
    parser.add_argument("--num_radial", type=int, default=32)
    parser.add_argument("--edge_sh_lmax", type=int, default=2)
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def load_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {path}")
    structures = read(path, index=":")
    if not structures:
        raise ValueError(f"No structures found in {path}")
    for frame_index, atoms in enumerate(structures):
        if "becs" not in atoms.arrays:
            raise KeyError(f"Frame {frame_index} has no per-atom 'becs' array")
        shape = np.asarray(atoms.arrays["becs"]).shape
        expected = (len(atoms), 9)
        if shape != expected:
            raise ValueError(
                f"Frame {frame_index}: becs shape is {shape}; expected {expected}"
            )
    return structures


def predict(atoms, model, args, device):
    with torch.no_grad():
        values = Calculate(
            atoms,
            model,
            graph_max_radius=args.graph_max_radius,
            num_radial=args.num_radial,
            edge_sh_lmax=args.edge_sh_lmax,
            radial_basis=None,
            device=device,
        )
    return values.detach().cpu().numpy().reshape(len(atoms), 9)


def metrics(errors):
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors**2))),
        float(np.mean(errors)),
    )


def main():
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    device = resolve_device(args.device)
    structures = load_dataset(args.test_xyz)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = torch.jit.load(str(args.model), map_location=device)
    model.eval()

    references = []
    predictions = []
    metadata = []
    for frame_index, atoms in enumerate(structures):
        reference = np.asarray(atoms.arrays["becs"], dtype=float)
        prediction = predict(atoms, model, args, device)
        references.append(reference[:, DIAGONAL_INDICES])
        predictions.append(prediction[:, DIAGONAL_INDICES])
        metadata.extend(
            (frame_index, atom_index, element)
            for atom_index, element in enumerate(
                atoms.get_chemical_symbols(), start=1
            )
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(structures):
            print(f"Predicted {frame_index + 1}/{len(structures)} structures")

    reference = np.concatenate(references)
    prediction = np.concatenate(predictions)
    error = reference - prediction

    with (args.output_dir / "bec_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame_index",
                "atom_index",
                "element",
                "component",
                "reference_bec",
                "predicted_bec",
                "abs_error",
            )
        )
        for row, (frame_index, atom_index, element) in enumerate(metadata):
            for component_index, component in enumerate(DIAGONAL_LABELS):
                ref = reference[row, component_index]
                pred = prediction[row, component_index]
                writer.writerow(
                    (
                        frame_index,
                        atom_index,
                        element,
                        component,
                        f"{ref:.8f}",
                        f"{pred:.8f}",
                        f"{abs(ref - pred):.8f}",
                    )
                )

    element_array = np.asarray([item[2] for item in metadata])
    with (args.output_dir / "metrics_by_element.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("element", "count", "mae", "rmse", "mean_signed_error"))
        for element in sorted(set(element_array)):
            selected = error[element_array == element].reshape(-1)
            mae, rmse, signed = metrics(selected)
            writer.writerow(
                (element, selected.size, f"{mae:.8f}", f"{rmse:.8f}", f"{signed:.8f}")
            )

    flat_reference = reference.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    flat_error = error.reshape(-1)
    mae, rmse, signed = metrics(flat_error)
    with (args.output_dir / "metrics_overall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("structures", "count", "mae", "rmse", "mean_signed_error"))
        writer.writerow(
            (len(structures), flat_error.size, f"{mae:.8f}", f"{rmse:.8f}", f"{signed:.8f}")
        )

    limit = max(
        5.0,
        float(
            np.ceil(
                max(np.max(np.abs(flat_reference)), np.max(np.abs(flat_prediction)))
            )
        ),
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(flat_reference, flat_prediction, s=5, alpha=0.35)
    axis.plot((-limit, limit), (-limit, limit), "--", color="black", linewidth=1)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Reference BEC (diagonal components)")
    axis.set_ylabel("Predicted BEC (diagonal components)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "bec_parity.png", dpi=200)
    plt.close(figure)

    print(f"Structures: {len(structures)}")
    print(f"MAE: {mae:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
