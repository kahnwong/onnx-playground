use ort::{GraphOptimizationLevel, Session, Tensor};
use ndarray::prelude::*;

fn main() -> ort::Result<()>{
    // init model
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../data/model-onnx/model.onnx")?;

    // prep data
    let data = vec![
        vec![5.7_f32, 3.0_f32, 4.2_f32, 1.2_f32],
        vec![6.3_f32, 3.3_f32, 4.7_f32, 1.6_f32],
        vec![6.7_f32, 3.1_f32, 5.6_f32, 2.4_f32],
        vec![5.9_f32, 3.0_f32, 5.1_f32, 1.8_f32],
        vec![6.7_f32, 3.1_f32, 4.7_f32, 1.5_f32],
        vec![6.6_f32, 3.0_f32, 4.4_f32, 1.4_f32],
    ];

    let data_ndarray: Array2<f32> = Array2::from_shape_vec((data.len(), data[0].len()), data.into_iter().flatten().collect())
        .expect("Invalid data");
    let tensor = Tensor::from_array(data_ndarray)?;
    // let tensor = tensor_raw.data_ptr_mut()?.cast::<f32>(); // for casting tensor dtype

    // prediction
    // let outputs = model.run([tensor.into()])?; // shorthand
    let outputs = model.run(ort::inputs!["X" => tensor]?)?;
    let predictions = outputs["output_label"].try_extract_tensor::<i64>()?;
    println!("{}", predictions);

    Ok(())
}
