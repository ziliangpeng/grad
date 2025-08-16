import random
import sys
from pathlib import Path
from micrograd import engine as mg

# Add the experimental directory to the Python path to allow sibling imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from grad import Variable


def run_tests():
    """Runs a series of tests to validate the autograd implementation against micrograd."""

    def validate_op(op_name, our_fn, micro_fn, *args):
        print(f"  Validating op: {op_name} with args {args}")
        # Our implementation
        our_vars = [Variable(value=val, requires_grad=True) for val in args]
        our_result = our_fn(*our_vars)
        our_result.backward()

        # Micrograd implementation
        micro_vars = [mg.Value(val) for val in args]
        micro_result = micro_fn(*micro_vars)
        micro_result.backward()

        # Validate forward and backward passes
        print(f"  Our result: {our_result.value}, Micro result: {micro_result.data}")
        assert (
            abs(our_result.value - micro_result.data) < 1e-6
        ), f"Forward pass mismatch for {op_name}"
        for i, (ours, micro) in enumerate(zip(our_vars, micro_vars)):
            assert (
                abs(ours.grad - micro.grad) < 1e-6
            ), f"Gradient mismatch for {op_name} on input {i}"
        print(f"  Validation successful for {op_name}.")

    print("Running validation against micrograd...")

    for i in range(20):
        print(f"Starting test iteration {i+1}/20...")
        a, b = random.uniform(-3, 3), random.uniform(-3, 3)

        # Test basic arithmetic operations
        validate_op("add", lambda x, y: x + y, lambda x, y: x + y, a, b)
        validate_op("mul", lambda x, y: x * y, lambda x, y: x * y, a, b)
        validate_op("sub", lambda x, y: x - y, lambda x, y: x - y, a, b)
        validate_op(
            "div", lambda x, y: x / y, lambda x, y: x / y, a, max(b, 0.1)
        )  # Avoid division by zero
        # validate_op("exp", lambda x: x.exp(), lambda x: x.exp(), a)

        # if a > 0:  # Power operation requires a positive base for log
            # validate_op("pow", lambda x, y: x ** y, lambda x, y: x ** y, a, b)

    print("✓ All validation tests passed.")


if __name__ == "__main__":
    run_tests()
