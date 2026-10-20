"""Fetch pinned OpenFAST/reference sources and build with CMake and GNU Fortran."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from windpower.simulation import CONTROLLER_SOURCE, OPENFAST_COMMIT, RTEST_COMMIT


def commands(prefix, compiler, jobs, cmake_args):
    for name, commit in (("openfast", OPENFAST_COMMIT), ("r-test", RTEST_COMMIT)):
        folder = prefix / name
        if not (folder / ".git").exists():
            yield ["git", "init", str(folder)]
        yield ["git", "-C", str(folder), "fetch", "--depth", "1", f"https://github.com/OpenFAST/{name}.git", commit]
        yield ["git", "-C", str(folder), "checkout", "--detach", commit]
    yield [
        "cmake", "-S", str(prefix / "openfast"), "-B", str(prefix / "build"),
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_Fortran_COMPILER={compiler}",
        "-DBUILD_TESTING=OFF", "-DDOUBLE_PRECISION=ON", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", *cmake_args,
    ]
    yield ["cmake", "--build", str(prefix / "build"), "--target", "openfast", "turbsim", "--parallel", str(jobs)]
    yield [
        compiler, "-shared", "-fPIC", "-cpp", "-DIMPLICIT_DLLEXPORT", "-ffree-line-length-none",
        "-fdefault-real-8", "-O2", str(prefix / "r-test" / CONTROLLER_SOURCE), "-o", str(prefix / "DISCON.dll"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("vendor"))
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for command in commands(args.prefix.resolve(), args.compiler, args.jobs, args.cmake_arg):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
