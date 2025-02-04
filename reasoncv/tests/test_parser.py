from reasoncv.parser import RcvOpParser
from reasoncv.ops import ZoomOp, ObjectDetectionOp, DepthEstimationOp


def test_parse_op_str():
    parser = RcvOpParser()
    op_contents = """
function: zoom
center: [0.7, 0.3]
size: [0.5, 0.3]
"""
    op_request = parser.parse_op_str(op_contents)
    assert isinstance(op_request, ZoomOp)
    assert op_request.center == (0.7, 0.3)
    assert op_request.size == (0.5, 0.3)


def test_object_detection_op():
    """Test proper ObjectDetectionOp parsing."""
    parser = RcvOpParser()
    op_contents = """
function: object-detection
object_name: car
threshold: 0.6
"""
    op_request = parser.parse_op_str(op_contents)
    assert isinstance(op_request, ObjectDetectionOp)
    assert op_request.object_name == "car"
    assert op_request.threshold == 0.6


def test_depth_estimation_op():
    """Test proper DepthEstimationOp parsing."""
    parser = RcvOpParser()
    op_contents = """
function: depth-estimation
"""
    op_request = parser.parse_op_str(op_contents)
    assert isinstance(op_request, DepthEstimationOp)


def test_zoom_missing_field():
    """Test that missing required ZoomOp field returns None."""
    parser = RcvOpParser(quiet=True)  # quiet=True to avoid raising an error
    op_contents = """
function: zoom
center: [0.7, 0.3]
"""  # Missing 'size'
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None


def test_zoom_wrong_type():
    """Test that providing wrong type for ZoomOp returns None."""
    parser = RcvOpParser(quiet=True)  # quiet=True to avoid raising an error
    op_contents = """
function: zoom
center: "not-a-tuple"
size: [0.5, 0.3]
"""
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None

    op_contents = """
function: zoom
center: 0.8
size: [0.5, 0.3]
"""
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None


def test_object_detection_missing_field():
    """Test that missing required ObjectDetectionOp field returns None."""
    parser = RcvOpParser(quiet=True)  # quiet=True to avoid raising an error
    op_contents = """
function: object-detection
object_name: tree
# Missing 'threshold'
"""
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None


def test_object_detection_wrong_type():
    """Test that providing wrong type for ObjectDetectionOp returns None."""
    parser = RcvOpParser(quiet=True)  # quiet=True to avoid raising an error
    op_contents = """
function: object-detection
object_name: dog
threshold: "high"
"""
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None


def test_zoom_extra_field():
    """
    Test that supplying an extra field for ZoomOp does not include it in the parsed model.
    By default, Pydantic ignores extra fields.
    """
    parser = RcvOpParser(quiet=True)  # quiet=True to avoid raising an error
    op_contents = """
function: zoom
center: [0.7, 0.3]
size: [0.5, 0.3]
unexpected_field: 42
"""
    op_request = parser.parse_op_str(op_contents)
    assert isinstance(op_request, ZoomOp)
    # Ensure that the extra field is not present in the model's dict
    assert "unexpected_field" not in op_request.dict()


def test_missing_function_field():
    """Test that YAML missing the required 'function' field returns None."""
    op_contents = """
object_name: car
threshold: 0.6
"""
    parser = RcvOpParser(quiet=True)
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None


def test_invalid_yaml_format():
    """Test that an invalid YAML format returns None."""
    parser = RcvOpParser(quiet=True)  # quiet=True to avoid raising an error
    op_contents = "just a string without YAML mapping"
    op_request = parser.parse_op_str(op_contents)
    assert op_request is None 