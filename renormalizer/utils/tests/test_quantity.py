# -*- coding: utf-8 -*-

import pytest
from pytest import approx
from renormalizer.utils import Quantity
import numpy as np
import math


def test_quantity_basic_creation():
    """Test basic Quantity object creation with different units"""
    # Test creation with different units
    q_au = Quantity(1.0, "a.u.")
    assert q_au.value == 1.0
    assert q_au.unit == "a.u."
    
    q_ev = Quantity(1.0, "eV")
    assert q_ev.value == 1.0
    assert q_ev.unit == "eV"
    
    q_cm = Quantity(1000, "cm^{-1}")
    assert q_cm.value == 1000
    assert q_cm.unit == "cm^{-1}"
    
    q_k = Quantity(300, "K")
    assert q_k.value == 300
    assert q_k.unit == "K"


def test_quantity_unit_conversion():
    """Test unit conversion functionality"""
    # Test conversion between different units
    q_au = Quantity(1.0, "a.u.")
    q_ev = q_au.as_unit("eV")
    q_cm = q_au.as_unit("cm^{-1}")
    
    # Check that conversion is approximately correct
    assert q_ev.value == approx(27.2114, rel=1e-4)
    assert q_cm.value == approx(2.1947e5, rel=1e-4)
    
    # Test round-trip conversion
    q_roundtrip = q_ev.as_unit("a.u.")
    assert q_roundtrip.value == approx(1.0, rel=1e-10)
    
    # Test array conversion
    q_array = Quantity([1.0, 2.0], "a.u.")
    q_array_ev = q_array.as_unit("eV")
    assert q_array_ev.values[0] == approx(27.2114, rel=1e-4)
    assert q_array_ev.values[1] == approx(54.4228, rel=1e-4)

    # test time conversion
    q3 = Quantity(1, "ps")
    np.testing.assert_allclose(q3.as_au(), 41341.37333518)


def test_quantity_arithmetic_operations():
    """Test arithmetic operations with Quantity objects"""
    q1 = Quantity(2.0, "a.u.")
    q2 = Quantity(3.0, "a.u.")
    
    # Test addition
    result_add = q1 + q2
    assert result_add.as_au() == approx(5.0)
    
    # Test subtraction
    result_sub = q1 - q2
    assert result_sub.as_au() == approx(-1.0)
    
    # Test multiplication with scalar
    result_mul = q1 * 2.0
    assert result_mul.as_au() == approx(4.0)
    
    # Test right multiplication
    result_rmul = 2.0 * q1
    assert result_rmul.as_au() == approx(4.0)
    
    # Test division
    result_div = q1 / 2.0
    assert result_div.as_au() == approx(1.0)
    
    # Test negation
    result_neg = -q1
    assert result_neg.as_au() == approx(-2.0)


def test_quantity_comparison_operations():
    """Test comparison operations with Quantity objects"""
    q1 = Quantity(2.0, "a.u.")
    q2 = Quantity(2.0, "a.u.")
    q3 = Quantity(3.0, "a.u.")
    
    # Test equality
    assert q1 == q2
    assert q1 != q3
    
    # Test comparison with zero
    assert q1 != 0
    zero_q = Quantity(0.0, "a.u.")
    assert zero_q == 0


def test_quantity_array_operations():
    """Test array operations with Quantity objects"""
    # Test creation with array
    values = [1.0, 2.0, 3.0]
    q_array = Quantity(values, "a.u.")
    
    # Test length
    assert len(q_array) == 3
    
    # Test indexing
    assert q_array[0].value == 1.0
    assert q_array[1].value == 2.0
    assert q_array[2].value == 3.0
    
    # Test slicing
    q_slice = q_array[0:2]
    assert len(q_slice) == 2
    assert q_slice[0].value == 1.0
    assert q_slice[1].value == 2.0
    
    # Test iteration
    for i, q in enumerate(q_array):
        assert q.value == values[i]
    
    # Test shape property
    assert q_array.shape == (3,)


def test_quantity_numpy_compatibility():
    """Test compatibility with numpy operations"""
    # Test with numpy scalar types
    np_float = np.float64(2.0)
    q = Quantity(3.0, "a.u.")
    
    result1 = np_float * q
    assert isinstance(result1, Quantity)
    assert result1.as_au() == approx(6.0)
    
    result2 = q * np_float
    assert isinstance(result2, Quantity)
    assert result2.as_au() == approx(6.0)
    
    # Test with numpy complex types
    # NOTE: We do not currently use complex numbers within Quantity
    # np_complex = np.complex128(1.0 + 2.0j)
    # result3 = np_complex * q
    # assert isinstance(result3, Quantity)
    # assert result3.as_au() == approx(3.0 + 6.0j)
    
    # Test array conversion
    q_array = Quantity([1.0, 2.0], "a.u.")
    np_array = np.array(q_array)
    assert np.allclose(np_array, [1.0, 2.0])


def test_quantity_to_beta():
    """Test conversion to beta (1/kT)"""
    # Test with temperature in K
    temp_k = Quantity(300, "K")
    beta = temp_k.to_beta()
    expected_beta = 1.0 / temp_k.as_au()
    assert beta == approx(expected_beta)
    
    # Test with zero temperature
    temp_zero = Quantity(0, "K")
    beta_zero = temp_zero.to_beta()
    assert beta_zero == math.inf
    
    # Test with array temperatures
    temps = Quantity([100, 200, 300], "K")
    betas = temps.to_beta()
    assert len(betas) == 3
    assert betas[0] == approx(1.0 / Quantity(100, "K").as_au())
    assert betas[1] == approx(1.0 / Quantity(200, "K").as_au())
    assert betas[2] == approx(1.0 / Quantity(300, "K").as_au())


def test_quantity_edge_cases():
    """Test edge cases and error conditions"""
    # Test with negative values
    q_neg = Quantity(-1.0, "a.u.")
    assert q_neg.value == -1.0
    
    # Test with very small values
    q_small = Quantity(1e-10, "a.u.")
    assert q_small.value == 1e-10
    
    # Test with very large values
    q_large = Quantity(1e10, "a.u.")
    assert q_large.value == 1e10
    
    # Test error for invalid unit
    with pytest.raises(ValueError):
        Quantity(1.0, "invalid_unit")
    
    # Test error for multiplication between quantities
    q1 = Quantity(1.0, "a.u.")
    q2 = Quantity(2.0, "a.u.")
    with pytest.raises(TypeError):
        _ = q1 * q2
    
    # Test error for division between quantities
    with pytest.raises(TypeError):
        _ = q1 / q2


def test_quantity_string_representation():
    """Test string representation of Quantity objects"""
    # Test scalar representation
    q_scalar = Quantity(1.5, "a.u.")
    assert str(q_scalar) == "1.5 a.u."
    assert repr(q_scalar) == "Quantity(1.5, 'a.u.')"
    
    # Test array representation
    q_array = Quantity([1.0, 2.0], "cm^{-1}")
    assert "[1.0, 2.0] cm^{-1}" in str(q_array)
    assert "Quantity([1.0, 2.0], 'cm^{-1}')" in repr(q_array)


def test_quantity_backward_compatibility():
    """Test backward compatibility with existing code"""
    # Test that .value works for scalars
    q_scalar = Quantity(1.0, "a.u.")
    assert q_scalar.value == 1.0
    
    # Test that .value raises error for arrays
    q_array = Quantity([1.0, 2.0], "a.u.")
    with pytest.raises(AttributeError):
        _ = q_array.value
    
    # Test that .values works for both scalars and arrays
    # For scalars, it returns a single-element array
    assert np.array_equal(q_scalar.values, [1.0])
    # For arrays, it returns the full array
    assert np.array_equal(q_array.values, [1.0, 2.0])
    
    # Test shape property - scalar has shape (1,) not ()
    assert q_scalar.shape == (1,)
    assert q_array.shape == (2,)


def test_quantity_mixed_operations():
    """Test operations between scalar and array quantities"""
    q_scalar = Quantity(2.0, "a.u.")
    q_array = Quantity([1.0, 3.0], "a.u.")
    
    # Test addition
    result_add = q_scalar + q_array
    assert np.allclose(result_add.as_au(), [3.0, 5.0])
    
    result_add2 = q_array + q_scalar
    assert np.allclose(result_add2.as_au(), [3.0, 5.0])
    
    # Test subtraction
    result_sub = q_scalar - q_array
    assert np.allclose(result_sub.as_au(), [1.0, -1.0])
    
    result_sub2 = q_array - q_scalar
    assert np.allclose(result_sub2.as_au(), [-1.0, 1.0])


if __name__ == "__main__":
    # Run all tests
    test_quantity_basic_creation()
    test_quantity_unit_conversion()
    test_quantity_arithmetic_operations()
    test_quantity_comparison_operations()
    test_quantity_array_operations()
    test_quantity_numpy_compatibility()
    test_quantity_to_beta()
    test_quantity_edge_cases()
    test_quantity_string_representation()
    test_quantity_backward_compatibility()
    test_quantity_mixed_operations()
