"""CPU-only checks for the BindingDB preparation example and inference inputs."""

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_bindingdb", ROOT / "examples/prepare_bindingdb.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

# Exercise the real data functions without loading GPU/model dependencies.
names = {"load_preprocessed_train_json", "load_molecule_data", "find_icl_dti", "get_columns"}
tree = ast.parse((ROOT / "es_merging_inference.py").read_text())
functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
inference = {"json": json, "np": np, "Chem": Chem, "AllChem": AllChem, "TanimotoSimilarity": TanimotoSimilarity}
exec(compile(functions, "es_merging_inference.py", "exec"), inference)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root / "raw", self.root / "prepared"
        self.source.mkdir()
        self.train = pd.DataFrame([
            ("CCO", "ACDE", 1),
            ("CCO", "FGHI", 0),
            ("CCN", "KLMN", 1),
        ], columns=prepare.COLUMNS)
        self.test = pd.DataFrame([("CCO", "PQRS", 0)], columns=prepare.COLUMNS)
        self.save_sources()

    def save_sources(self):
        self.train.to_csv(self.source / "train.csv", index=False)
        self.test.to_csv(self.source / "test.csv", index=False)

    def run_preparation(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return prepare.prepare_dataset(self.source, self.output, workers=2, **kwargs)

    def test_round_trip_and_icl_with_shared_ligand(self):
        report = self.run_preparation()
        self.assertEqual(report["unique_molecules"], 2)
        output_csv = pd.read_csv(self.output / "test.csv")
        pd.testing.assert_frame_equal(output_csv, self.test)
        pd.testing.assert_frame_equal(pd.read_csv(self.output / "train.csv"), self.train)
        with redirect_stdout(io.StringIO()):
            train = inference["load_preprocessed_train_json"](self.output / "train_3d.json")
            molecules = inference["load_molecule_data"](self.output / "test_3d.json")
        self.assertEqual(molecules["CCO"]["coordinates"].shape, (3, 3))
        self.assertEqual(train[0]["fingerprint_obj"].GetNumBits(), 2048)
        self.assertEqual(train[0]["Target"], "ACDE")
        self.assertEqual(train[0]["Y"], 1)
        for train_embeddings, test_embeddings in [(None, None),
                ({"ACDE": np.ones(2), "FGHI": np.ones(2), "KLMN": np.ones(2)}, {"ACDE": np.ones(2)})]:
            examples = inference["find_icl_dti"]("ACDE", "CCO", "BindingDB_protein", train,
                                                   train_embeddings, test_embeddings, top_k=3)
            self.assertEqual(len(examples), 3)
            self.assertEqual(examples[0]["label"], "Interacts")
            self.assertEqual(examples[1]["protein"], "FGHI")
            self.assertEqual(examples[1]["label"], "Does not interact")

    def test_invalid_smiles_is_removed_consistently_and_reported(self):
        self.train.loc[3] = ("invalid_smiles", "ACDE", 1)
        self.test.loc[1] = ("invalid_smiles", "PQRS", 0)
        self.save_sources()
        report = self.run_preparation()
        self.assertEqual(report["failed_molecules"], 1)
        self.assertEqual(report["splits"]["test"]["written"], 1)
        failures = pd.read_csv(self.output / "failed_rows.csv")
        self.assertEqual(set(failures["split"]), {"train", "test"})
        self.assertEqual(set(failures["Ligand"]), {"invalid_smiles"})
        for split in ("train", "test"):
            records = json.loads((self.output / f"{split}_3d.json").read_text())
            csv_data = pd.read_csv(self.output / f"{split}.csv")
            self.assertNotIn("invalid_smiles", set(csv_data["Ligand"]))
            self.assertEqual({r["Drug"] for r in records}, set(csv_data["Ligand"]))
            self.assertTrue(all(r["atoms"] and r["coordinates"] for r in records))

    def test_invalid_labels_and_missing_sequences_are_rejected(self):
        for column, value in [("classification_label", 0.5), ("classification_label", None), ("Protein", ""), ("Protein", "P12345")]:
            with self.subTest(column=column, value=value):
                frame = self.train.astype(object)
                frame.loc[0, column] = value
                frame.to_csv(self.source / "bad.csv", index=False)
                with self.assertRaises(ValueError):
                    prepare.load_split(self.source / "bad.csv")
        self.train.drop(columns="Ligand").to_csv(self.source / "bad.csv", index=False)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            prepare.load_split(self.source / "bad.csv")

    def test_reproducible_coordinates_and_atom_alignment(self):
        smiles = "C[C@H](O)c1ccccc1"
        first, second = prepare.generate_molecule(smiles, 42), prepare.generate_molecule(smiles, 42)
        np.testing.assert_array_equal(first["coordinates"], second["coordinates"])
        self.assertEqual(first["atoms"], [a.GetSymbol() for a in Chem.MolFromSmiles(smiles).GetAtoms()])
        self.assertTrue(np.isfinite(first["coordinates"]).all())
        self.assertEqual(len("".join(first["fingerprint"])), 2048)

    def test_no_2d_or_empty_coordinate_fallback(self):
        with patch.object(prepare.AllChem, "EmbedMolecule", return_value=-1) as embed:
            with self.assertRaisesRegex(ValueError, "embedding failed"):
                prepare.generate_molecule("CCO", 42)
            self.assertEqual(embed.call_count, 2)

    def test_source_protection_and_subset_limit(self):
        hashes = {p.name: prepare.file_sha256(p) for p in self.source.glob("*.csv")}
        report = self.run_preparation(max_rows=1)
        self.assertEqual(report["splits"]["train"]["written"], 1)
        self.assertEqual(report["sources"]["train"]["total_rows"], 3)
        self.assertEqual(hashes, {p.name: prepare.file_sha256(p) for p in self.source.glob("*.csv")})
        with self.assertRaises(FileExistsError):
            self.run_preparation()
        with self.assertRaises(ValueError):
            prepare.prepare_dataset(self.source, self.source, overwrite=True)

    def test_download_checksum_and_cached_sources(self):
        data = b"Ligand,Protein,classification_label\nCCO,ACDE,1\n"
        sources = {"train": ("public-file-id", hashlib.sha256(data).hexdigest())}
        download_dir = self.root / "download"
        with patch.object(prepare, "DOWNLOADS", sources), patch.object(prepare, "urlopen", return_value=io.BytesIO(data)) as fetch:
            with redirect_stdout(io.StringIO()):
                prepare.download_splits(download_dir)
                prepare.download_splits(download_dir)
            self.assertEqual(fetch.call_count, 1)
        with patch.object(prepare, "DOWNLOADS", sources), patch.object(prepare, "urlopen", return_value=io.BytesIO(b"<html>download failed</html>")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare.download_splits(self.root / "bad_download")
            self.assertFalse((self.root / "bad_download/train.csv").exists())
            self.assertFalse((self.root / "bad_download/train.csv.part").exists())


if __name__ == "__main__":
    unittest.main()
