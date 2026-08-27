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

DEMOS = {
    "hashmap": hashmap.demo,
    "sketch": sketch.demo,
    "record": record.demo,
    "sequence": sequence.demo,
    "ngram": ngram.demo,
    "graph": graph.demo,
    "fsm": fsm.demo,
    "sdm": sdm.demo,
    "field": field.demo,
    "attribute": attribute_field.demo,
    "spatial": spatial.demo,
    "phase": phase.demo,
    "crdt": crdt.demo,
    "fit": fit.demo,
    "render": render.demo,
    "color": color.demo,
    "orset": orset.demo,
    "codec": phase.demo_codec,
    "turntable": color.demo_turntable,
    "dispatch": dispatch.demo,
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
