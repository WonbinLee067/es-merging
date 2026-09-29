"""Prepare the PSICHIC BindingDB/protein example for es_merging_inference.py.

Proteins remain amino-acid sequences; only molecules receive 3D conformers.
Source: https://github.com/huankoh/PSICHIC/blob/main/dataset/README.md
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from tqdm import tqdm


# Public files linked from the PSICHIC dataset release, BindingDB/protein.
DOWNLOADS = {
    "train": (
        "1Lyogc3ZptZyDhaxO904nXYFpDgdxejwR",
        "038e9f9e4a0773768c0753e5c49c1597a0475e983eb3c64fcb8ebb7970d9288d",
    ),
    "test": (
        "1AwF5wQ463Ba4hJf2zRxBTayShGqbyuRw",
        "a02d849944d5a712f6f28d0597b6b0a8ae43b2b80b3b3f23a8a6d372885e7b23",
    ),
}
COLUMNS = ["Ligand", "Protein", "classification_label"]
OUTPUT_FILES = [
    "train.csv", "test.csv", "train_3d.json", "test_3d.json",
    "failed_rows.csv", "preprocessing_report.json",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_splits(input_dir):
    """Download the published split without replacing existing source files."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for split, (file_id, expected_hash) in DOWNLOADS.items():
        target = input_dir / f"{split}.csv"
        if not target.exists():
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Downloading {split}.csv from PSICHIC", flush=True)
            with tempfile.TemporaryDirectory(dir=input_dir, prefix=".download-") as directory:
                temporary = Path(directory) / target.name
                with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                if file_sha256(temporary) != expected_hash:
                    raise ValueError(f"Download checksum mismatch for {split}.csv; check the source link.")
                temporary.replace(target)
        if file_sha256(target) != expected_hash:
            raise ValueError(f"{target} differs from the published split. Use another input directory for --download.")
        print(f"Verified {target}", flush=True)


def load_split(path, max_rows=None):
    frame = pd.read_csv(path, dtype={"Ligand": str, "Protein": str})
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    total_rows = len(frame)
    frame = frame.loc[:, COLUMNS].head(max_rows) if max_rows is not None else frame.loc[:, COLUMNS]
    frame = frame.copy()
    for column in ("Ligand", "Protein"):
        frame[column] = frame[column].str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"{path}: {column} contains missing or empty values")
    if not frame["Protein"].str.fullmatch(r"[A-Z]+").all():
        raise ValueError(f"{path}: Protein must contain amino-acid sequences, not IDs or structure paths")
    labels = pd.to_numeric(frame["classification_label"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path}: classification_label must be binary (0 or 1)")
    frame["classification_label"] = labels.astype(int)
    if frame.empty:
        raise ValueError(f"{path}: no input rows")
    return frame, total_rows


def generate_molecule(smiles, seed):
    """Generate one conformer and an ICL fingerprint, retaining SMILES atom order."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty SMILES")
    expected_atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    parameters.maxIterations = 1000
    if AllChem.EmbedMolecule(molecule, parameters) < 0:
        parameters.useRandomCoords = True
        if AllChem.EmbedMolecule(molecule, parameters) < 0:
            raise ValueError("3D embedding failed after retry")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
    molecule = Chem.RemoveHs(molecule)
    atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    coordinates = molecule.GetConformer().GetPositions()
    if atoms != expected_atoms:
        raise ValueError("Atom order changed during hydrogen removal")
    if coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid 3D coordinates")
    return {
        "Drug": smiles,
        "atoms": atoms,
        "coordinates": coordinates.tolist(),
        # The inference loader concatenates a list of binary strings.
        "fingerprint": [fingerprint.ToBitString()],
    }


def prepare_dataset(input_dir, output_dir, *, max_rows=None, seed=42, workers=4, overwrite=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Use separate input and output directories to preserve the original splits")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    if workers < 1 or not 0 <= seed < 2**31:
        raise ValueError("workers must be positive and seed must be in [0, 2**31)")
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Existing output files: {existing}. Choose another output directory or use --overwrite.")

    frames, sources = {}, {}
    for split in DOWNLOADS:
        path = input_dir / f"{split}.csv"
        frame, total_rows = load_split(path, max_rows)
        frames[split] = frame
        sources[split] = {"path": str(path), "sha256": file_sha256(path), "total_rows": total_rows}
    smiles_list = list(dict.fromkeys(smiles for frame in frames.values() for smiles in frame["Ligand"]))
    molecules, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_molecule, smiles, seed): smiles for smiles in smiles_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Molecule 3D"):
            smiles = futures[future]
            try:
                molecules[smiles] = future.result()
            except Exception as error:
                errors[smiles] = f"{type(error).__name__}: {error}"

    failures, cleaned, records, counts = [], {}, {}, {}
    for split, frame in frames.items():
        valid = frame["Ligand"].isin(molecules)
        for index, row in frame.loc[~valid].iterrows():
            failures.append({"split": split, "csv_row": int(index) + 2, "Ligand": row["Ligand"], "error": errors[row["Ligand"]]})
        cleaned[split] = frame.loc[valid]
        counts[split] = {"selected": len(frame), "written": int(valid.sum()), "failed": int((~valid).sum())}
        if split == "train":
            records[split] = [
                {**molecules[row.Ligand], "Target": row.Protein, "Y": int(row.classification_label)}
                for row in cleaned[split].itertuples(index=False)
            ]
        else:
            records[split] = [
                {key: molecules[smiles][key] for key in ("Drug", "atoms", "coordinates")}
                for smiles in dict.fromkeys(cleaned[split]["Ligand"])
            ]

    report = {
        "sources": sources, "seed": seed, "max_rows_per_split": max_rows,
        "rdkit_version": rdBase.rdkitVersion, "conformer_method": "ETKDGv3 with MMFF94/UFF when available",
        "unique_molecules": len(smiles_list), "failed_molecules": len(errors), "splits": counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["split", "csv_row", "Ligand", "error"]).to_csv(output_dir / "failed_rows.csv", index=False)
    (output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(frame.empty for frame in cleaned.values()):
        raise ValueError(f"No usable rows in one or more splits; see {output_dir / 'failed_rows.csv'}")
    for split in frames:
        cleaned[split].to_csv(output_dir / f"{split}.csv", index=False)
        with (output_dir / f"{split}_3d.json").open("w") as handle:
            json.dump(records[split], handle, allow_nan=False)
            handle.write("\n")
    print(json.dumps(counts, indent=2))
    print(f"Prepared files in {output_dir}. Failed rows are recorded in failed_rows.csv.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("./data/BindingDB_protein/raw"), help="Directory containing the source train.csv and test.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("./data/BindingDB_protein"))
    parser.add_argument("--download", action="store_true", help="Download/verify the public PSICHIC protein split in input_dir")
    parser.add_argument("--max_rows", type=int, default=None, help="Use only the first N rows of each split for a small example")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files in output_dir")
    args = parser.parse_args()
    if args.download:
        download_splits(args.input_dir)
    prepare_dataset(args.input_dir, args.output_dir, max_rows=args.max_rows,
                    seed=args.seed, workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
