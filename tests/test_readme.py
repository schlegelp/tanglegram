"""Check that the examples in the README actually run.

The snippets are executed in order against a shared namespace, exactly as a
reader would work through them.
"""

import pathlib
import re

import matplotlib

matplotlib.use("Agg")

import pytest


README = pathlib.Path(__file__).parent.parent / "README.md"


def python_blocks(heading, until):
    """Every ```Python block between two headings of the README."""
    text = README.read_text()
    section = text[text.index(heading) : text.index(until)]
    return re.findall(r"```Python\n(.*?)```", section, re.S)


def test_examples_run():
    pytest.importorskip("plotly")

    blocks = python_blocks("### Dendrograms", "### Untangling methods")
    assert len(blocks) >= 12, "expected the worked examples to still be there"

    namespace = {}
    for i, block in enumerate(blocks, 1):
        # Opening a browser is not helpful in a test
        code = block.replace("fig.show()", "pass")
        try:
            exec(compile(code, f"<README block {i}>", "exec"), namespace)
        except Exception as exc:
            raise AssertionError(
                f"README example {i} failed: {exc}\n\n{block}"
            ) from exc


def test_readme_images_exist():
    """Every image the README points at must be committed alongside it."""
    text = README.read_text()
    local = re.findall(r'<img src="(?!https?://)([^"]+)"', text)
    assert local, "expected the README to reference local images"

    missing = [p for p in local if not (README.parent / p).is_file()]
    assert not missing, f"README references missing images: {missing}"
