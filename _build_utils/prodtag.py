#!/usr/bin/env python3
import os
import textwrap


def prod_tag():
    from time import gmtime, strftime
    import platform

    proddate = strftime("%y-%m-%d %H:%M:%S", gmtime())
    try:
        prodhost = platform.uname()
    except AttributeError:
        prodhost = ("???", "???", "???")

    return proddate, prodhost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", help="Save tag to this file")
    parser.add_argument(
        "--meson-dist",
        help="Output path is relative to MESON_DIST_ROOT",
        action="store_true",
    )
    args = parser.parse_args()

    date, host = prod_tag()

    # For NumPy 2.0, this should only have one field: `version`
    template = textwrap.dedent(
        f"""
        DATE = "{date}"
    """
    )

    if args.write:
        outfile = args.write
        if args.meson_dist:
            outfile = os.path.join(os.environ.get("MESON_DIST_ROOT", ""), outfile)

        # Print human readable output path
        relpath = os.path.relpath(outfile)
        if relpath.startswith("."):
            relpath = outfile

        with open(outfile, "w") as f:
            print(f"Saving prod date to {relpath}")
            f.write(template)
    else:
        print(date)
