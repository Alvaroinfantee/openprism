//! Streaming kernels. No fast-math, worker pool, global mutable state, or unsafe code.
use numpy::ndarray::{Array2, ArrayD, IxDyn};
use numpy::{PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn reflect(index: isize, length: usize) -> usize {
    if length == 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let folded = index.rem_euclid(period) as usize;
    if folded < length {
        folded
    } else {
        period as usize - folded
    }
}

/// Two sliding sums with O(width) scratch, no padded image or intermediate plane.
/// f64 accumulation bounds cancellation for the normalized [0, 1] input domain.
fn blur(source: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let size = 2 * radius + 1;
    let divisor = (size * size) as f64;
    let mut columns = vec![0.0_f64; width];
    for dy in -r..=r {
        let row = reflect(dy, height) * width;
        for (x, column) in columns.iter_mut().enumerate() {
            *column += f64::from(source[row + x]);
        }
    }
    let mut output = vec![0.0_f32; height * width];
    for y in 0..height {
        let mut sum = 0.0;
        for dx in -r..=r {
            sum += columns[reflect(dx, width)];
        }
        for x in 0..width {
            output[y * width + x] = (sum / divisor) as f32;
            if x + 1 < width {
                sum += columns[reflect(x as isize + r + 1, width)]
                    - columns[reflect(x as isize - r, width)];
            }
        }
        if y + 1 < height {
            let entering = reflect(y as isize + r + 1, height) * width;
            let leaving = reflect(y as isize - r, height) * width;
            for (x, column) in columns.iter_mut().enumerate() {
                *column += f64::from(source[entering + x]) - f64::from(source[leaving + x]);
            }
        }
    }
    output
}

#[pyfunction]
fn box_blur<'py>(
    py: Python<'py>,
    value: PyReadonlyArray2<'py, f32>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if !value.data().is_aligned() {
        return Err(PyValueError::new_err("expected aligned input"));
    }
    let array = value.as_array();
    let (height, width) = array.dim();
    if height == 0 || width == 0 || radius > 64 {
        return Err(PyValueError::new_err(
            "native blur requires nonempty input and radius <= 64",
        ));
    }
    let source = array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("expected C-contiguous input"))?;
    if source
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err(PyValueError::new_err(
            "native blur requires finite values in [0, 1]",
        ));
    }
    // Keep the GIL while borrowing NumPy memory; no input copy is needed.
    let output = blur(source, height, width, radius);
    let result = Array2::from_shape_vec((height, width), output)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
fn palette<'py>(
    py: Python<'py>,
    value: PyReadonlyArrayDyn<'py, f32>,
    positions: Vec<f32>,
    colors: Vec<[u8; 3]>,
    widths: Vec<f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    if !value.data().is_aligned() {
        return Err(PyValueError::new_err("expected aligned input"));
    }
    if positions.len() < 2
        || colors.len() != positions.len()
        || widths.len() + 1 != positions.len()
        || positions.iter().any(|v| !v.is_finite())
        || positions.windows(2).any(|v| v[0] >= v[1])
        || widths.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(PyValueError::new_err("invalid palette stops"));
    }
    let array = value.as_array();
    let source = array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("expected C-contiguous input"))?;
    let normalized: Vec<[f32; 3]> = colors
        .iter()
        .map(|c| c.map(|v| f32::from(v) / 255.0))
        .collect();
    let length = source
        .len()
        .checked_mul(3)
        .ok_or_else(|| PyValueError::new_err("image too large"))?;
    let mut output = vec![0_u8; length];
    for (value, pixel) in source.iter().zip(output.chunks_exact_mut(3)) {
        let v = value.clamp(0.0, 1.0);
        if v.is_nan() {
            continue;
        } // Reference masks select no interval for NaN.
        if v < positions[0] {
            pixel.copy_from_slice(&colors[0]);
            continue;
        }
        if v > positions[positions.len() - 1] {
            pixel.copy_from_slice(&colors[colors.len() - 1]);
            continue;
        }
        // At shared stops, the later interval wins, exactly as in the reference.
        for i in (0..widths.len()).rev() {
            if v >= positions[i] && v <= positions[i + 1] {
                let factor = ((v - positions[i]) / widths[i]).clamp(0.0, 1.0);
                for (c, out) in pixel.iter_mut().enumerate() {
                    let delta = normalized[i + 1][c] - normalized[i][c];
                    let color = normalized[i][c] + factor * delta;
                    *out = (color * 255.0).round_ties_even() as u8;
                }
                break;
            }
        }
    }
    let mut shape = array.shape().to_vec();
    shape.push(3);
    let result = ArrayD::from_shape_vec(IxDyn(&shape), output)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, result))
}

#[pymodule]
fn openprism_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("API_VERSION", 1)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(box_blur, module)?)?;
    module.add_function(wrap_pyfunction!(palette, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflected_edges_and_singleton() {
        assert_eq!(
            (-4..=4).map(|i| reflect(i, 3)).collect::<Vec<_>>(),
            vec![0, 1, 2, 1, 0, 1, 2, 1, 0]
        );
        assert_eq!(blur(&[0.5], 1, 1, 64), vec![0.5]);
    }

    #[test]
    fn sliding_matches_direct_f64_neighbourhood() {
        let (height, width) = (7, 11);
        let source: Vec<f32> = (0..height * width)
            .map(|i| (i % 13) as f32 / 13.0)
            .collect();
        for radius in [0, 1, 3, 15, 64] {
            let result = blur(&source, height, width, radius);
            let r = radius as isize;
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            sum += f64::from(
                                source[reflect(y as isize + dy, height) * width
                                    + reflect(x as isize + dx, width)],
                            );
                        }
                    }
                    assert_eq!(
                        result[y * width + x],
                        (sum / ((2 * radius + 1).pow(2) as f64)) as f32
                    );
                }
            }
        }
    }
}
