"""`simlens` command-line interface: inspect bundles, verify audit trails, run eval."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _info(args):
    from .bundle import Bundle

    b = Bundle.load(args.bundle)
    print(json.dumps(b._manifest(), indent=2))
    named = sum(1 for n in b.feature_names if n)
    print(f"named features: {named}/{b.n_features}")
    print(f"concepts: {b.concept_names}")


def _verify(args):
    from .bundle import Bundle

    b = Bundle.load(args.bundle)
    ok = b.verify()
    print(f"content hash: {'OK' if ok else 'MISMATCH'} ({b.content_hash})")
    if args.secret:
        sig = b.verify_signature(args.secret)
        print(f"signature: {'OK' if sig else 'INVALID'}")
    sys.exit(0 if ok else 1)


def _eval(args):
    from .explain import Explainer

    ex = Explainer(args.bundle)
    X = np.load(args.vectors)
    n = min(len(X), args.limit)
    pairs = [(X[i], X[i + 1]) for i in range(0, n - 1, 2)]
    from .eval import scorecard

    print(json.dumps(scorecard(ex, pairs), indent=2))


def _serve(args):
    from .serve import main as serve_main

    argv = ["--port", str(args.port)]
    if args.bundle:
        argv += ["--bundle", args.bundle]
    serve_main(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simlens", description="SimLens CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="print a bundle's manifest")
    pi.add_argument("bundle")
    pi.set_defaults(func=_info)

    pv = sub.add_parser("verify", help="verify a bundle's hash (and signature)")
    pv.add_argument("bundle")
    pv.add_argument("--secret", default=None)
    pv.set_defaults(func=_verify)

    pe = sub.add_parser("eval", help="faithfulness scorecard over a vectors .npy file")
    pe.add_argument("bundle")
    pe.add_argument("vectors", help="path to an .npy array [N, dim]")
    pe.add_argument("--limit", type=int, default=40)
    pe.set_defaults(func=_eval)

    ps = sub.add_parser("serve", help="start the HTTP serving sidecar")
    ps.add_argument("--bundle", default=None)
    ps.add_argument("--port", type=int, default=8008)
    ps.set_defaults(func=_serve)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
