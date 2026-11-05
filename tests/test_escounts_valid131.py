"""CPU engineering checks, not benchmark predictions or model-performance tests."""
import ast
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np
import torch
from scipy import ndimage

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import escounts_valid131_common as c

SOURCE_VALUE = os.environ.get("ESCOUNTS_SOURCE")
SOURCE = Path(SOURCE_VALUE).resolve() if SOURCE_VALUE else None
requires_source = unittest.skipUnless(
    SOURCE is not None and SOURCE.is_dir(),
    "Official-source integration check requires ESCOUNTS_SOURCE pointing to the pinned checkout",
)


def test_temp_root():
    """Use only an explicitly authorized temporary path, independent of layout."""
    explicit = os.environ.get("ESCOUNTS_TEST_TMP")
    if explicit:
        root = c.storage_path(explicit)
    else:
        root = None
        for key in ("TMP", "TEMP"):
            value = os.environ.get(key)
            if value:
                try:
                    root = c.storage_path(value)
                    break
                except ValueError:
                    pass
        if root is None:
            raise unittest.SkipTest("Set ESCOUNTS_TEST_TMP or TMP to an E: / authorized project temporary directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


# This helper is not itself a pytest test.
test_temp_root.__test__ = False


def official_method(file, cls, method):
    tree = ast.parse((SOURCE / file).read_text(encoding="utf-8"))
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    return next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == method)


class Interfaces(unittest.TestCase):
    @requires_source
    def test_full_valid_population(self):
        rows = c.valid_rows(SOURCE)
        self.assertEqual(len(rows), 131)
        zero = [r for r in rows if float(r["count"]) == 0]
        self.assertEqual([r["name"] for r in zero], ["stu9_67.mp4"])
        self.assertTrue(all(float(r["num_frames"]) > 64 for r in rows))

    @requires_source
    def test_actual_valid_density_against_official_ast(self):
        fn = official_method("Repcount_multishot_loader.py", "Rep_count", "__getitem__")
        # Execute the original density block unchanged; exclude later empty argmin.
        nodes, active = [], False
        for node in fn.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "frame_ids" for t in node.targets):
                active = True
            if active:
                nodes.append(node)
            if active and isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "count" for t in node.targets):
                break
        self.assertTrue(nodes)
        code = compile(ast.Module(nodes, []), "<unaltered official density block>", "exec")
        for row in c.valid_rows(SOURCE):
            env = dict(np=np, math=math, ndimage=ndimage,
                self=SimpleNamespace(density_peak_width=0.5),
                num_frames=int(float(row["num_frames"])), segment_start=int(float(row["segment_start"])),
                segment_end=int(float(row["segment_end"])), segment_id=0,
                lim_constraint=np.inf, cycle=c.cycle_points(row))
            exec(code, env)
            expected = torch.from_numpy(env["gt_density"]).half()
            actual, total = c.density_target(row)
            with self.subTest(name=row["name"]):
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(total, expected.sum()))
                self.assertTrue(torch.isfinite(total))

    @requires_source
    def test_two_offsets_slicing_and_pooling(self):
        # Test arrays have 768 channels but are not videos or model predictions.
        data = np.arange(11 * 768 * 2 * 3 * 3, dtype=np.float32).reshape(11, 768, 2, 3, 3)
        row = dict(name="interface_fixture", segment_start="8", segment_end="80")
        actual = c.two_offset_tokens(data, row)
        import einops
        # Execute the complete UNMODIFIED official load_tokens method with real
        # Torch/NumPy/einops. Its test role is the contract's two-offset reference;
        # no train/test dataset, CSV or real media is opened.
        fn = official_method("Repcount_multishot_loader.py", "Rep_count", "load_tokens")
        env = dict(np=np, torch=torch, einops=einops, math=math)
        exec(compile(ast.Module([fn], []), "<unaltered official load_tokens>", "exec"), env)
        obj = SimpleNamespace(split="test", pool_tokens=.4, select_rand_segment=False)
        root = test_temp_root()
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path = Path(folder) / "interface_only.npz"
            np.savez(path, data)
            reference = env["load_tokens"](obj, str(path), False, bounds=(8, 80), get_overlapping_segments=True)
        for result, expected in zip(actual, reference):
            self.assertTrue(torch.equal(result, expected))
        self.assertEqual(tuple(c.batch_window(actual[0], "cpu").shape), (1, actual[0].shape[1] * 4, 768))

    @requires_source
    def test_original_zero_shot_branch_ignores_yi(self):
        fn = official_method("video_mae_cross_full_attention.py", "SupervisedMAE", "forward")
        branch = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "shot_num > 0")
        code = compile(ast.Module([branch], []), "<unaltered official zero-shot branch>", "exec")
        def forbidden(_):
            raise AssertionError("yi was read")
        obj = SimpleNamespace(decoder_embed=forbidden, shot_token=torch.arange(24.).reshape(3, 8))
        values = []
        for yi in (None, torch.zeros(1, 1, 768), torch.full((1, 19, 768), float("nan"))):
            env = dict(self=obj, shot_num=0, yi=yi, x=torch.zeros(1, 8, 8))
            exec(code, env)
            values.append(env["y"])
        self.assertTrue(all(torch.equal(values[0], x) for x in values[1:]))

    def test_native_metrics_and_padding(self):
        p1, p2 = torch.ones(1, 8), torch.ones(1, 4)
        combined = c.merge_predictions(p1, p2)
        self.assertTrue(torch.equal(combined[:, :4], torch.full((1, 4), .5)))
        with self.assertRaises(ValueError):
            c.merge_predictions(torch.ones(1, 9), p2)
        rows = [dict(name="zero", rounded_count=2., raw_count=1.5, native_target=0., csv_count=0.),
                dict(name="positive", rounded_count=4., raw_count=4.5, native_target=3.5, csv_count=4.)]
        scores = c.score_records(rows)
        q, g = torch.tensor([2., 4.]), torch.tensor([0., 3.5])
        expected = ((q - g).abs() / (g + .1)).numpy().mean()
        self.assertEqual(scores["native_VALID131"]["MAE_native"], float(expected))
        self.assertEqual(scores["paperform_CSV_positive130"]["n"], 1)
        self.assertTrue(scores["zero_count_rows"][0]["false_positive"])
        self.assertTrue(torch.equal(torch.round(torch.tensor([.5, 1.5, 2.5])), torch.tensor([0., 2., 2.])))

    def test_path_rejects_train_and_test(self):
        for name in ("train", "test"):
            with self.assertRaisesRegex(ValueError, "Forbidden train/test"):
                c.data_path(test_temp_root() / name / "some.npz")

    def test_write_scope_and_verified_resume(self):
        if os.name == "nt":
            with self.assertRaises(ValueError):
                c.storage_path("C:/not-authorized/output.json")
        root = test_temp_root()
        with tempfile.TemporaryDirectory(dir=root) as folder:
            token = Path(folder) / "valid.npz"
            token.write_bytes(b"interface-only fixture, not a real feature")
            expected = dict(contract=c.CONTRACT, source_commit=c.COMMIT, encoder_sha256=c.ENCODER_SHA,
                            video_sha256="fixture", name="fixture.mp4")
            with self.assertRaises(ValueError):
                c.reusable_token(token, expected)
            receipt = {**expected, "sha256": c.sha256(token)}
            c.write_json(token.with_suffix(".npz.receipt.json"), receipt)
            self.assertEqual(c.reusable_token(token, expected), receipt)
            with self.assertRaises(ValueError):
                c.reusable_token(token, {**expected, "video_sha256": "changed"})
            token.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                c.reusable_token(token, expected)

    @requires_source
    def test_actual_model_import(self):
        # Missing dependency is a visible skip, never a synthetic forward success.
        try:
            c.load_official(SOURCE)
        except ModuleNotFoundError as exc:
            self.skipTest(f"Official full model import unavailable: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
