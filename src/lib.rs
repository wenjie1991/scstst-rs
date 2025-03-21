use pyo3::prelude::*;
use numpy::PyArray3;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use indicatif::ProgressBar;
use num_cpus;
use std::time::Instant;

/// co_occurrence function
#[pyfunction]
fn co_occur_count(py: Python, v_x: Vec<f64>, v_y: Vec<f64>, v_radium: Vec<f64>, v_label: Vec<i32>) -> PyResult<PyObject> {

    let chunk_size = 1000;
    //let thread_num = 5;
    let thread_num = num_cpus::get();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_num)
        .build()
        .unwrap();


    let start = Instant::now();
    let pb = ProgressBar::new(v_x.len() as u64);

    // Unique labels and indices mapping
    let label_unique: HashSet<_> = v_label.iter().cloned().collect();
    let mut v_unique_label: Vec<_> = label_unique.into_iter().collect();
    v_unique_label.sort_unstable(); // Ensure a consistent order for indexing
    let label_map: HashMap<i32, usize> = v_unique_label.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    let k = v_unique_label.len();
    let l = v_radium.len();

    // Precompute squared radii
    let v_radium_sq: Vec<f64> = v_radium.iter().map(|x| x.powi(2)).collect();


    // Store coordinates for better cache locality
    let points: Vec<(f64, f64, usize)> = v_x.into_iter()
        .zip(v_y.into_iter())
        .zip(v_label.into_iter())
        .map(|((x, y), label)| (x, y, *label_map.get(&label).unwrap()))
        .collect();

    // Use an atomic vector to avoid mutex locking
    let co_occur_m = Arc::new((0..(k * k * l)).map(|_| AtomicI32::new(0)).collect::<Vec<_>>());

    // Parallel computation
    pool.install(|| {
        points.par_chunks(chunk_size).for_each(|chunk| {

            let mut local_counts = vec![0; k * k * l]; // Thread-local storage
            for (i, &(x_i, y_i, label_i)) in chunk.iter().enumerate() {

                for (j, &(x_j, y_j, label_j)) in points.iter().enumerate() {
                    if i != j {
                        let distance = (x_i - x_j).powi(2) + (y_i - y_j).powi(2);

                        for r in 0..l {
                            if distance <= v_radium_sq[l - r - 1] {
                                let index = r * k * k + label_i * k + label_j;
                                local_counts[index] += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }

            }
            // Merge results into shared atomic vector
            for (idx, &count) in local_counts.iter().enumerate() {
                if count > 0 {
                    co_occur_m[idx].fetch_add(count, Ordering::Relaxed);
                }
            }
            pb.inc(chunk.len() as u64);
        });
    });

    pb.finish_with_message("Finished co-occurrence computation");
    println!("Elapsed time: {:?}", start.elapsed());

    // Convert to a normal Vec<i32> for NumPy conversion
    let co_occur_m_final: Vec<i32> = co_occur_m.iter().map(|x| x.load(Ordering::Relaxed)).collect();

    // Reshape into a 3D vector
    let co_occur_reshaped: Vec<Vec<Vec<i32>>> = (0..k)
        .map(|i| (0..k)
            .map(|j| (0..l)
                .map(|r| co_occur_m_final[r * k * k + i * k + j])
                .collect()
            ).collect()
        ).collect();

    // Convert to NumPy array
    let array = PyArray3::from_vec3(py, &co_occur_reshaped).unwrap();

    Ok(array.into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn scstat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(co_occur_count, m)?)?;
    Ok(())
}

