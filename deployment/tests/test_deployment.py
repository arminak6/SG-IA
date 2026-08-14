from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

import common  # noqa: E402
import import_state  # noqa: E402
import validate_deployment  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_relative_path_rejects_absolute_and_traversal_paths(self) -> None:
        for value in ("../secret.txt", "/absolute.txt", "folder/../../escape.txt"):
            with self.subTest(value=value):
                with self.assertRaises(common.DeploymentError):
                    common.relative_path(value, field="path")

    def test_load_manifest_verifies_hash_and_safe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            material = root / "material"
            material.mkdir()
            document = material / "example.txt"
            document.write_text("portable corpus", encoding="utf-8")
            digest = hashlib.sha256(document.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_root": "material",
                        "documents": [
                            {
                                "path": "example.txt",
                                "wiki_path": "department/example.txt",
                                "sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(common, "ROOT", root):
                loaded = common.load_manifest(manifest)
            self.assertEqual(len(loaded.documents), 1)
            self.assertEqual(loaded.documents[0].sha256, digest)

    def test_load_manifest_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            material = root / "material"
            material.mkdir()
            (material / "example.txt").write_text("changed", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_root": "material",
                        "documents": [
                            {
                                "path": "example.txt",
                                "wiki_path": "example.txt",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(common, "ROOT", root):
                with self.assertRaisesRegex(common.DeploymentError, "Hash mismatch"):
                    common.load_manifest(manifest)


class RestoreSafetyTests(unittest.TestCase):
    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                data = b"must not escape"
                member = tarfile.TarInfo("../outside.txt")
                member.size = len(data)
                handle.addfile(member, io.BytesIO(data))
            with self.assertRaisesRegex(common.DeploymentError, "Unsafe archive path"):
                import_state.safe_extract(archive, root / "destination")
            self.assertFalse((root / "outside.txt").exists())

    def test_safe_extract_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "link.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("link")
                member.type = tarfile.SYMTYPE
                member.linkname = "target"
                handle.addfile(member)
            with self.assertRaisesRegex(common.DeploymentError, "Unsupported archive member"):
                import_state.safe_extract(archive, root / "destination")


class StartupValidationTests(unittest.TestCase):
    def test_placeholder_credential_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            credentials = root / "credentials.json"
            credentials.write_text(
                json.dumps(
                    {
                        "aws_access_key_id": "",
                        "aws_secret_access_key": "",
                    }
                ),
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "SGIA_AWS_CREDENTIALS_FILE=./credentials.json\n",
                encoding="utf-8",
            )
            with (
                patch.object(validate_deployment, "ROOT", root),
                patch.dict(validate_deployment.os.environ, {}, clear=True),
            ):
                with self.assertRaisesRegex(common.DeploymentError, "placeholders"):
                    validate_deployment.validate_credentials()

    def test_structurally_valid_credential_file_is_accepted_without_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            credentials = root / "credentials.json"
            credentials.write_text(
                json.dumps(
                    {
                        "aws_access_key_id": "A" * 20,
                        "aws_secret_access_key": "s" * 40,
                    }
                ),
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "SGIA_AWS_CREDENTIALS_FILE=./credentials.json\n",
                encoding="utf-8",
            )
            with (
                patch.object(validate_deployment, "ROOT", root),
                patch.dict(validate_deployment.os.environ, {}, clear=True),
            ):
                result = validate_deployment.validate_credentials()
            self.assertEqual(result, {"configured": True, "source": "read_only_json"})
            self.assertNotIn("A" * 20, repr(result))


if __name__ == "__main__":
    unittest.main()
