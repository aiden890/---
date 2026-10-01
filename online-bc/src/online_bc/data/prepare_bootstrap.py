import argparse
import subprocess
from pathlib import Path
from online_bc.learning.learner_service import unpack

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--transport-python", required=True)
p.add_argument("--token-file", required=True)
a = p.parse_args()
root = Path(a.root)
for model in ["pi05", "xiaomi"]:
    incoming = root / "incoming" / model
    subprocess.run(
        [
            a.transport_python,
            "-m",
            "online_bc.transport.hf_transfer",
            "download",
            str(incoming),
            "pi05-cup-online-bc-20261001/bootstrap/" + model,
            "--token-file",
            a.token_file,
        ],
        check=True,
    )
    unpack(incoming, root / "data" / model)
