"""Test that generate functions are called with correct arguments for different model types."""

from unittest.mock import MagicMock, patch

import pytest


def test_generate_mamba_not_called_with_images_arg():
    """Test that generate_mamba is called without images argument.

    This is a regression test for issue #220 where generate_mamba() was called
    with a positional `images` argument that it doesn't accept, causing:
    TypeError: generate_mamba() takes 2 positional arguments but 3 were given
    """
    from mistral_inference.generate import generate_mamba

    # Verify generate_mamba signature doesn't accept images as positional arg
    import inspect

    sig = inspect.signature(generate_mamba)
    params = list(sig.parameters.keys())

    # First two are positional: encoded_prompts, model
    # After that should be keyword-only (indicated by * in signature)
    assert params[0] == "encoded_prompts"
    assert params[1] == "model"

    # Check that 'images' is NOT in the parameters at all
    assert "images" not in params, "generate_mamba should not accept images parameter"

    # Verify max_tokens is keyword-only
    max_tokens_param = sig.parameters.get("max_tokens")
    assert max_tokens_param is not None
    assert max_tokens_param.kind == inspect.Parameter.KEYWORD_ONLY


def test_generate_accepts_images_arg():
    """Test that generate() accepts images as positional argument."""
    from mistral_inference.generate import generate

    import inspect

    sig = inspect.signature(generate)
    params = list(sig.parameters.keys())

    # Verify images is the 3rd positional parameter
    assert params[0] == "encoded_prompts"
    assert params[1] == "model"
    assert params[2] == "images"

    # images should be positional-or-keyword (not keyword-only)
    images_param = sig.parameters.get("images")
    assert images_param is not None
    assert images_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
