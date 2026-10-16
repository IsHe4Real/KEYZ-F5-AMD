import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from keyz_f5_amd import assets
from keyz_f5_amd.engine import validate_request


class PackageTests(unittest.TestCase):
    def test_valid_request(self):
        self.assertEqual(validate_request({'reply_id':'trial-01','text':'Ready.'}),('trial-01','Ready.'))

    def test_invalid_request(self):
        for value in (None,[],{}, {'reply_id':'../escape','text':'Hello'},
                      {'reply_id':'ok','text':'a'*181}, {'reply_id':'ok','text':'你好'},
                      {'reply_id':'ok','text':''}):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_request(value)

    def test_import_copies_and_verifies_without_changing_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); sources=[]; expected={}
            for name in assets.HASHES:
                path=root/name; path.write_bytes(name.encode()); sources.append(path)
                expected[name]=hashlib.sha256(name.encode()).hexdigest()
            with patch.object(assets,'HASHES',expected):
                assets.import_assets(*sources, root/'bundle')
                self.assertEqual(assets.verify_assets(root/'bundle'),expected)
                for source in sources: self.assertEqual(source.read_bytes(),(root/'bundle'/source.name).read_bytes())
                with self.assertRaises(FileExistsError): assets.import_assets(*sources,root/'bundle')

    def test_bad_asset_rejected_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wrong';path.write_bytes(b'wrong')
            with self.assertRaises(ValueError): assets.import_assets(path,path,path,path,Path(folder)/'bundle')
            self.assertFalse((Path(folder)/'bundle').exists())

    def test_missing_assets_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'asset'): assets.verify_assets(folder)

    def test_empty_and_mutated_asset_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vocab.txt';path.write_bytes(b'wrong')
            with patch.object(assets,'HASHES',{'vocab.txt':hashlib.sha256(b'correct').hexdigest()}):
                with self.assertRaises(ValueError): assets.verify_assets(folder)

    def test_metadata_import_has_no_engine_initialization(self):
        import keyz_f5_amd
        self.assertEqual(keyz_f5_amd.__version__,'0.1.0a1')


if __name__=='__main__': unittest.main()
