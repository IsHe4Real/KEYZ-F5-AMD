"""Pinned model checks and explicit local bundle import. No automatic downloads."""
import hashlib
from pathlib import Path
import shutil

HASHES = {
    'F5_Preprocess.onnx':'e71d3ed14e90ba3fc1e83560512e86a771e25b6f0be7789b2cf53a5c9ba5617d',
    'F5_Transformer.onnx':'3ffc57282d7bd645a6c20abcc0db2ea8e1ae547c4ccdd8a025eb2620fb7fd929',
    'F5_Decode.onnx':'a16fea891beb4889b47e5987b80c841bb996beaccc1d1fe27a1f9323a089a6da',
    'vocab.txt':'2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_assets(folder):
    for name, expected in HASHES.items():
        path = Path(folder)/name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or incompatible model asset: '+name)
    return dict(HASHES)


def import_assets(preprocess, transformer, decode, vocab, destination):
    sources = dict(zip(HASHES, map(Path, (preprocess, transformer, decode, vocab))))
    for name, source in sources.items():
        if digest(source) != HASHES[name]:
            raise ValueError('Incompatible source: '+name)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing bundle.
    for name, source in sources.items():
        shutil.copyfile(source, destination/name)
    return verify_assets(destination)
