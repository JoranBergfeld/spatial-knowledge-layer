import pytest
from pydantic import ValidationError

from spatial_core.records import Value, ValueProvenance


def test_a_derived_value_has_to_name_its_rule() -> None:
    with pytest.raises(ValidationError):
        Value(value="floor", provenance=ValueProvenance.DERIVED)

    assert Value.derived("floor", "floor-from-level-tags@1").rule == "floor-from-level-tags@1"


def test_an_inferred_value_has_to_record_its_basis() -> None:
    with pytest.raises(ValidationError):
        Value(value="office", provenance=ValueProvenance.INFERRED)


def test_a_sourced_value_needs_neither() -> None:
    assert Value.sourced("107").provenance is ValueProvenance.SOURCED
