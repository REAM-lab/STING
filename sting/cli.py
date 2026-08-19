import argparse
import runpy
from importlib.resources import files


EXAMPLES = {
    "9_bus_emt": "emt_simulation/wscc_9/run.py",
    "9_bus_ssm": "small_signal_models/wscc_9/run.py",
    "9_bus_mor": "model_order_reduction/wscc_9/run.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run STING examples."
    )

    parser.add_argument(
        "example",
        choices=EXAMPLES,
        help="Example to run",
    )

    args = parser.parse_args()

    example = files("examples") / EXAMPLES[args.example]

    runpy.run_path(str(example), run_name="__main__")