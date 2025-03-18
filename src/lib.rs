use pyo3::prelude::*;
use numpy::PyArray3;
use std::collections::HashSet;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// co_occurence function
#[pyfunction]
fn co_occur_count(py: Python, v_x: Vec<f64>, v_y: Vec<f64>, v_radium: Vec<f64>, v_label: Vec<i32>) -> PyResult<PyObject> {
    let n = v_x.len();

        // Unique labels and indices
    let label_unique: HashSet<_> = v_label.iter().cloned().collect();
    let v_unique_label: Vec<_> = label_unique.into_iter().collect();
    let k = v_unique_label.len();

    // Precompute squared radii
    let v_radium_sq: Vec<f64> = v_radium.iter().map(|x| x.powi(2)).collect();
    let l = v_radium_sq.len();

    // Shared co-occurrence matrix with Mutex for safe parallel writes
    let co_occur_m = Arc::new(Mutex::new(vec![0; k * k * l]));


    // Parallel computation
    (0..n).into_par_iter().for_each(|i| {
        let mut local_counts = vec![0; k * k * l]; // Thread-local storage

        for j in 0..n {
            if i != j {
                let distance = (v_x[i] - v_x[j]).powi(2) + (v_y[i] - v_y[j]).powi(2);

                for r in 0..l {
                    let radium_sq = v_radium_sq[l - r - 1];

                    if distance <= radium_sq {
                        let index = r * k * k + (v_label[i] as usize) * k + (v_label[j] as usize);
                        local_counts[index] += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        // Merge results into shared matrix (thread-safe)
        let mut co_occur_m_lock = co_occur_m.lock().unwrap();
        for i in 0..co_occur_m_lock.len() {
            co_occur_m_lock[i] += local_counts[i];
        }
    });

    // Convert to numpy array
    let co_occur_m_final = co_occur_m.lock().unwrap();
    let co_occur_reshaped: Vec<Vec<Vec<i32>>> = (0..k)
        .map(|i| (0..k)
            .map(|j| (0..l)
                .map(|r| co_occur_m_final[r * k * k + i * k + j])
                .collect()
            ).collect()
        ).collect();

    let array = PyArray3::from_vec3(py, &co_occur_reshaped).unwrap();

    //let dict = PyDict::new(py);
    //dict.set_item("co_occur", array)?;
    //Ok(dict.into())
    Ok(array.into())
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn scstat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(co_occur_count, m)?)?;
    Ok(())
}
