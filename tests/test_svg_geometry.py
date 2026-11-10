"""The SVG geometry behind band cuts.

Typst can't fit a 1600px-tall diagram on one page, so it is sliced into horizontal bands. Where the
slice lands is decided by an ink profile built from the SVG's own geometry — no rasteriser, no
dependencies. If that profile misreads a shape, the cut goes through a node box and the PDF shows a
label chopped in half. QA reports the weight it saw, but only if the geometry is right, so the
geometry is what has to be tested directly.
"""

from __future__ import annotations

from ba_agent.artifacts.typst import (
    IDENTITY_TRANSFORM,
    _apply_transform,
    _crop_svg,
    _ink_profile,
    _path_points,
    _shape_bboxes,
    svg_native_size,
)


def _svg(body: str, width: float = 600, height: float = 400) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>'
    )


def test_transforms_compose_in_order():
    """A nested group's transform is its parent's transform followed by its own."""
    nested = _apply_transform("translate(10, 20)", _apply_transform("translate(5, 5)", IDENTITY_TRANSFORM))
    assert nested[4] == 15.0 and nested[5] == 25.0, nested

    scaled_then_moved = _apply_transform("scale(2) translate(3, 4)", (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    # Scale first, then the translation lands in scaled units.
    assert scaled_then_moved[0] == 2.0
    assert scaled_then_moved[5] == 8.0

    rotated = _apply_transform("rotate(90)", (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    assert abs(rotated[0]) < 1e-9
    assert abs(rotated[1] - 1.0) < 1e-9


def test_unparseable_transform_is_ignored_not_crashed():
    assert _apply_transform("translate(x, y)", (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert _apply_transform(None, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert _apply_transform("weird(1,2)", (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def test_path_commands_are_read_in_absolute_and_relative_form():
    assert _path_points("M10 10 L30 50") == [(10.0, 10.0), (30.0, 50.0)]
    # H and V fix one axis only.
    assert _path_points("M10 10 H40 V30") == [(10.0, 10.0), (40.0, 10.0), (40.0, 30.0)]
    # Lower-case h/v are relative to the current point.
    assert _path_points("M10 10 h20 v15") == [(10.0, 10.0), (30.0, 10.0), (30.0, 25.0)]
    # Cubic: control points included, endpoint last - enough for a y-extent.
    assert _path_points("M0 0 C10 20 30 40 50 60") == [(0.0, 0.0), (10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
    # Quadratic and shorthand curves.
    assert _path_points("M0 0 Q10 20 30 40") == [(0.0, 0.0), (10.0, 20.0), (30.0, 40.0)]
    # Arc: only the endpoint matters for a y-extent.
    assert _path_points("M0 0 A10 10 0 0 1 40 80") == [(0.0, 0.0), (40.0, 80.0)]
    assert _path_points("") == []
    assert _path_points("M") == []


def test_rect_circle_line_text_and_polygon_all_produce_boxes():
    svg = _svg(
        '<rect x="10" y="100" width="120" height="30"/>'
        '<circle cx="50" cy="200" r="20"/>'
        '<line x1="0" y1="250" x2="100" y2="260"/>'
        '<polygon points="20,300 80,300 80,330 20,330"/>'
        '<text y="360" font-size="16">label</text>'
    )
    boxes = _shape_bboxes(svg)
    assert len(boxes) == 5

    def spans(y):
        return [b for b in boxes if b[0] <= y <= b[1]]

    assert spans(115)[0][2] == 3, "a node rect must count as box weight"
    assert spans(200)[0][2] == 3, "a circle is a node too"
    assert spans(255)[0][2] == 1, "a bare connector may be crossed"
    assert spans(315)[0][2] == 3
    assert spans(350)[0][2] == 3, "a label above its baseline must still be protected"


def test_group_transform_moves_a_shape_before_it_is_scanned():
    svg = _svg('<g transform="translate(0,150)"><rect x="10" y="0" width="50" height="20"/></g>')
    y_min, y_max, weight = _shape_bboxes(svg)[0]
    assert (y_min, y_max, weight) == (150.0, 170.0, 3)

    svg_scaled = _svg('<g transform="scale(2)"><rect x="10" y="50" width="10" height="10"/></g>')
    assert _shape_bboxes(svg_scaled)[0][:2] == (100.0, 120.0)

    # A closed group must not leave the stack unbalanced for the shapes after it.
    both = _svg(
        '<g transform="translate(0,100)"><rect x="0" y="0" width="10" height="10"/></g>'
        '<rect x="0" y="300" width="10" height="10"/>'
    )
    assert [b[:2] for b in _shape_bboxes(both)] == [(100.0, 110.0), (300.0, 310.0)]


def test_ink_profile_rows_match_the_viewbox_height():
    svg = _svg('<rect x="0" y="100" width="600" height="40"/>', height=400)
    profile = _ink_profile(svg, 400)
    assert len(profile) == 400
    assert profile[120] == 3
    assert profile[300] == 0

    # A smaller row count still maps the shape onto the right band of rows.
    coarse = _ink_profile(svg, 100)
    assert sum(1 for row in coarse if row == 3) == 11, "y 100..140 of 400 is rows 25..35 inclusive"
    assert coarse[24] == 0 and coarse[36] == 0


def test_ink_profile_without_a_viewbox_is_empty_not_a_crash():
    assert _ink_profile("<svg><rect x='0' y='0' width='10' height='10'/></svg>", 100) == []
    assert _ink_profile("", 100) == []
    assert _ink_profile(_svg("<rect/>"), 0) == []


def test_crop_rewrites_the_viewbox_and_the_root_size_together():
    svg = _svg('<rect x="0" y="200" width="600" height="50"/>', height=400)
    cropped = _crop_svg(svg, 190.0, 260.0)
    assert 'viewBox="0.0 190.0 600.0 70.0"' in cropped
    assert 'height="70.0"' in cropped
    assert 'width="600.0"' in cropped

    # Without a viewBox there is nothing to rewrite; returning the input unchanged is the safe answer.
    assert _crop_svg("<svg><rect/></svg>", 1, 2) == "<svg><rect/></svg>"

    # A root element with no explicit width gets one added, otherwise Typst guesses the aspect ratio.
    bare = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400"><rect/></svg>'
    cut = _crop_svg(bare, 100.0, 150.0)
    assert 'width="600.0"' in cut and 'height="50.0"' in cut


def test_native_size_is_read_before_banding_decisions():
    assert svg_native_size(_svg("")) == (600.0, 400.0)
    assert svg_native_size('<svg viewBox="0 0 800 1200"><rect/></svg>') == (800.0, 1200.0)
    assert svg_native_size("<svg></svg>") is None
