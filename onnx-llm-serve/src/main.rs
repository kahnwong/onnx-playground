use ort::{GraphOptimizationLevel, Session, inputs};
use tokenizers::Tokenizer;
use ndarray::{array, concatenate, s, Array1, ArrayViewD, Axis};
use rand::Rng;

const PROMPT: &str = "Who are you";
const GEN_TOKENS: i32 = 90;
const TOP_K: usize = 5;

fn main() -> ort::Result<()>{
    // init
    let mut rng = rand::thread_rng();

    // init model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("/home/kahnwong/Downloads/gpt2.onnx")?;
        // .commit_from_file("/home/kahnwong/Git/installer/Phi-3-mini-128k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32/phi3-mini-128k-instruct-cpu-int4-rtn-block-32.onnx")?;

    // Load the tokenizer and encode the prompt into a sequence of tokens.
    // let tokenizer = Tokenizer::from_file("/home/kahnwong/Git/installer/Phi-3-mini-128k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32/tokenizer.json").unwrap();
    let tokenizer = Tokenizer::from_file("/home/kahnwong/Downloads/tokenizer.json").unwrap();
    let tokens = tokenizer.encode(PROMPT, false).unwrap();
    let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

    let mut tokens = Array1::from_iter(tokens.iter().cloned());

    for _ in 0..GEN_TOKENS {
        let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let outputs = session.run(inputs![array]?)?;
        let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_tensor()?;

        // Collect and sort logits
        let probabilities = &mut generated_tokens
            .slice(s![0, 0, -1, ..])
            .insert_axis(Axis(0))
            .to_owned()
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>();
        probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        // Sample using top-k sampling
        let token = probabilities[rng.gen_range(0..=TOP_K)].0;
        tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

        let token_str = tokenizer.decode(&[token as _], true).unwrap();
        print!("{}", token_str);
    }

    Ok(())
}
