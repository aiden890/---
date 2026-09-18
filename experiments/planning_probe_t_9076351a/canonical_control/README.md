# Generic Qwen semantic-plan canonicalization gate (t_8c1f107f)

This offline probe separates model decisions from trusted registry materialization. The model emits only `{"plan":[{"name":...,"args":...}]}`. The local canonicalizer validates every name, exact argument key/value, step count, and uniqueness before adding `instruction`, `contract`, and `budget`. It never selects or repairs a sequence.

The gate uses the three preserved reset contact sheets and four semantically equivalent task instructions with official `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`, greedy direct generation, and no parser repair, constrained decoding, or fallback.

Run on an idle amp_csi RTX3090 with one model process:

    python3 canonical_probe.py --images /parent_results/images --out /output

Verify locally:

    python3 -m unittest experiments/planning_probe_t_9076351a/canonical_control/test_canonical.py
    python3 -m py_compile experiments/planning_probe_t_9076351a/canonical_control/*.py
    (cd experiments/planning_probe_t_9076351a/canonical_control/results && sha256sum -c artifacts.sha256)

The hard gate passes only when direct semantic JSON, exact three-skill sequence, and canonicalized strict five-field plan are each 12/12. Production code is outside this directory and is not modified.
