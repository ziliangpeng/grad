import random
import sys
from pathlib import Path
import torch
# from micrograd import engine as mg

# Add the experimental directory to the Python path to allow sibling imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from grad import Variable

EPS = 1e-6


def run_tests():
    """Runs a series of tests to validate the autograd implementation against torch."""

    def validate_op(op_name, our_fn, torch_fn, *args):
        print(f"  Validating op: {op_name} with args {args}")
        # Our implementation
        our_vars = [Variable(value=val, requires_grad=True) for val in args]
        our_result = our_fn(*our_vars)
        our_result.backward()

        # PyTorch implementation
        torch_vars = [torch.tensor(float(val), requires_grad=True) for val in args]
        torch_result = torch_fn(*torch_vars)
        torch_result.backward()

        # Validate forward and backward passes
        print(f"  Our result: {our_result.value}, Torch result: {torch_result.item()}")
        print(f"  diff: {abs(our_result.value - torch_result.item())}")
        forward_diff = abs(our_result.value - torch_result.item())
        forward_scale = max(1.0, abs(our_result.value), abs(torch_result.item()))
        assert (
            forward_diff < EPS * forward_scale
        ), f"Forward pass mismatch for {op_name}"
        for i, (ours, torch_var) in enumerate(zip(our_vars, torch_vars)):
            print(f"  Our grad: {ours.grad}, Torch grad: {torch_var.grad.item()}")
            grad_diff = abs(ours.grad - torch_var.grad.item())
            grad_scale = max(1.0, abs(ours.grad), abs(torch_var.grad.item()))
            assert (
                grad_diff < EPS * grad_scale
            ), f"Gradient mismatch for {op_name} on input {i}. Ours: {ours.grad}, Torch: {torch_var.grad.item()}; diff: {grad_diff}"
        print(f"  Validation successful for {op_name}.")

    print("Running validation against torch...")

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
        validate_op("exp", lambda x: x.exp(), lambda x: x.exp(), a)

        if a > 0:  # Power operation requires a positive base for log
            validate_op("pow", lambda x, y: x ** y, lambda x, y: x ** y, a, b)

        c, d, e, f = random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3)
        # make sure d (w) is positive to simplify pow.
        validate_op("combo1", lambda x, y, z, w, v: x * y + z ** w / v, lambda x, y, z, w, v: x * y + z ** w / v, a, b, abs(c), d, e)

    print("✓ All validation tests passed.")


if __name__ == "__main__":
    run_tests()
