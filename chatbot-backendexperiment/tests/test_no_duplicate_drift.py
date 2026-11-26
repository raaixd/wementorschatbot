"""The duplicated deployment files (KB, index.html, logo) must stay identical."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sync_duplicates.py"


def _load():
    spec = importlib.util.spec_from_file_location("sync_duplicates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DuplicateDriftTests(unittest.TestCase):
    def test_copies_match_canonical(self):
        drift = _load().find_drift()
        self.assertEqual(
            drift, [], f"Duplicated files drifted; run `python scripts/sync_duplicates.py`: {drift}"
        )

    def test_detects_drift(self):
        import tempfile

        mod = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for canonical, copies in mod.DUPLICATES.items():
                for rel in [canonical, *copies]:
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    (root / rel).write_bytes(b"same")
            self.assertEqual(mod.find_drift(root), [])
            (root / "api/wementors_kb.json").write_bytes(b"changed")
            self.assertEqual(mod.find_drift(root), [("knowledge/wementors_kb.json", "api/wementors_kb.json")])


if __name__ == "__main__":
    unittest.main()
