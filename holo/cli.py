"""SDK demo runner, installed as the `hdc-demos` console script.

    hdc-demos                 # every demo, capacity tables + figures
    hdc-demos fsm graph       # a subset
    hdc-demos --dim 16384     # bigger hypervectors
"""

import argparse

from . import (
    attribute_field,
    color,
    crdt,
    dispatch,
    field,
    fit,
    fsm,
    graph,
    hashmap,
    ngram,
    orset,
    phase,
    record,
    render,
    sdm,
    sequence,
    sketch,
    spatial,
)
from .resonator import demo as resonator_demo

DEMOS = {
    "attribute": attribute_field.demo,
    "codec": phase.demo_codec,
    "color": color.demo,
    "crdt": crdt.demo,
    "dispatch": dispatch.demo,
    "field": field.demo,
    "fit": fit.demo,
    "fsm": fsm.demo,
    "graph": graph.demo,
    "hashmap": hashmap.demo,
    "ngram": ngram.demo,
    "orset": orset.demo,
    "phase": phase.demo,
    "record": record.demo,
    "render": render.demo,
    "resonator": resonator_demo,
    "sdm": sdm.demo,
    "sequence": sequence.demo,
    "sketch": sketch.demo,
    "spatial": spatial.demo,
    "turntable": color.demo_turntable,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", choices=[*DEMOS, []],
                    help=f"which demos (default: all): {', '.join(DEMOS)}")
    ap.add_argument("--dim", type=int, default=4096,
                    help="hypervector dimensionality (default 4096)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for name in args.names or DEMOS:
        DEMOS[name](dim=args.dim, seed=args.seed)


if __name__ == "__main__":
    main()
