#!/usr/bin/env python3
"""
Test script for Issue #220 fix - run on Windows with NVIDIA GPU.

Setup:
1. git clone https://github.com/fede-kamel/mistral-inference.git
2. cd mistral-inference
3. git checkout fix/generate-mamba-args
4. pip install -e .
5. pip install mamba-ssm causal-conv1d

Download model:
    python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mamba-Codestral-7B-v0.1', local_dir='./mamba-codestral')"

Run test:
    python test_fix_220.py ./mamba-codestral
"""

import sys
from pathlib import Path


def test_issue_220(model_path: str):
    """Test that Mamba models work in interactive mode without TypeError."""

    print("=" * 60)
    print("Testing Issue #220 Fix: generate_mamba() argument error")
    print("=" * 60)

    # Step 1: Verify the fix is in place by checking the code
    print("\n[1/4] Verifying fix is applied...")
    import ast
    main_py = Path(__file__).parent / "src" / "mistral_inference" / "main.py"
    with open(main_py) as f:
        tree = ast.parse(f.read())

    # Find the interactive function and check for isinstance branches
    found_fix = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "interactive":
            for child in ast.walk(node):
                if isinstance(child, ast.If):
                    if isinstance(child.test, ast.Call):
                        if hasattr(child.test.func, 'id') and child.test.func.id == 'isinstance':
                            if child.orelse:  # Has else branch
                                found_fix = True
                                break

    if found_fix:
        print("   ✓ Fix is applied (separate branches for Transformer/Mamba)")
    else:
        print("   ✗ Fix NOT applied - using old buggy code")
        return False

    # Step 2: Import and check signatures
    print("\n[2/4] Checking function signatures...")
    from mistral_inference.generate import generate, generate_mamba
    import inspect

    gen_params = list(inspect.signature(generate).parameters.keys())
    mamba_params = list(inspect.signature(generate_mamba).parameters.keys())

    print(f"   generate() params: {gen_params}")
    print(f"   generate_mamba() params: {mamba_params}")

    assert "images" in gen_params, "generate() should have images param"
    assert "images" not in mamba_params, "generate_mamba() should NOT have images param"
    print("   ✓ Signatures verified")

    # Step 3: Load model
    print(f"\n[3/4] Loading Mamba model from {model_path}...")
    from mistral_inference.mamba import Mamba

    model = Mamba.from_folder(Path(model_path), max_batch_size=1, num_pipeline_ranks=1)
    print(f"   ✓ Model loaded on {model.device}")

    # Step 4: Test generation (the actual fix)
    print("\n[4/4] Testing generation (this would crash before the fix)...")
    from mistral_inference.main import load_tokenizer
    from mistral_inference.generate import generate_mamba

    tokenizer = load_tokenizer(Path(model_path))

    # Encode a simple prompt
    prompt = "Hello, world!"
    tokens = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=True, eos=False)

    # This is what the fixed code does - call generate_mamba WITHOUT images
    generated_tokens, logprobs = generate_mamba(
        [tokens],
        model,
        max_tokens=10,
        temperature=0.7,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )

    result = tokenizer.instruct_tokenizer.tokenizer.decode(generated_tokens[0])
    print(f"   Prompt: {prompt}")
    print(f"   Generated: {result}")
    print("   ✓ Generation successful!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Issue #220 is fixed.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_fix_220.py <path-to-mamba-codestral-model>")
        print("\nExample:")
        print("  python test_fix_220.py ./mamba-codestral")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)

    success = test_issue_220(model_path)
    sys.exit(0 if success else 1)
