// // use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel};
// use onnxruntime::{environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel, session::Session};
// use tokenizers::Tokenizer;
// use std::error::Error;
// use std::fs;
// use std::io::{self, Write};

// fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
//     // Load the tokenizer
//     let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

//     // Load the ONNX model bytes
//     let model_bytes = fs::read("llama3_model.onnx")?;

//     // Create the environment
//     let environment = Environment::builder()
//         .with_name("llama3_inference")
//         .with_log_level(onnxruntime::LoggingLevel::Warning)
//         .build()?;

//     // Load the ONNX model from memory
//     let session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_model_from_memory(&model_bytes)?;

//     loop {
//         // Get the prompt from the user
//         print!("Enter your prompt: ");
//         io::stdout().flush()?;
//         let mut prompt = String::new();
//         io::stdin().read_line(&mut prompt)?;

//         // Tokenize the input
//         let encoding = tokenizer.encode(prompt.trim(), true)?;
//         let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
//         let input_tensor = vec![input_ids];

//         // Run the model
//         let outputs: Vec<OrtOwnedTensor<f32>> = session.run(vec![input_tensor.into()])?;

//         // Decode the output
//         let output_ids: Vec<i64> = outputs[0].iter().map(|&id| id as i64).collect();
//         let response = tokenizer.decode(output_ids, true)?;

//         // Print the response
//         println!("Response: {}", response);
//     }
// }



// use onnxruntime::{environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel};
// use tokenizers::Tokenizer;
// use std::error::Error;
// use std::fs;
// use std::io::{self, Write};

// fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
//     // Load the tokenizer
//     let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

//     // Load the ONNX model bytes
//     let model_bytes = fs::read("llama3_model.onnx")?;

//     // Create the environment
//     let environment = Environment::builder()
//         .with_name("llama3_inference")
//         .with_log_level(onnxruntime::LoggingLevel::Warning)
//         .build()?;

//     // Load the ONNX model from memory
//     let session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_model_from_memory(&model_bytes)?;

//     loop {
//         // Get the prompt from the user
//         print!("Enter your prompt: ");
//         io::stdout().flush()?;
//         let mut prompt = String::new();
//         io::stdin().read_line(&mut prompt)?;

//         // Tokenize the input
//         let encoding = tokenizer.encode(prompt.trim(), true)?;
//         let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
//         let input_tensor = Array::from_shape_vec((1, input_ids.len()), input_ids)?;

//         // Run the model
//         let outputs: Vec<OrtOwnedTensor<f32>> = session.run(vec![input_tensor.into()])?;

//         // Decode the output
//         let output_ids: Vec<i64> = outputs[0].iter().map(|&id| id as i64).collect();
//         let response = tokenizer.decode(output_ids, true)?;

//         // Print the response
//         println!("Response: {}", response);
//     }
// }




// use onnxruntime::{environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel};
// use tokenizers::Tokenizer;
// use std::error::Error;
// use std::fs;
// use std::io::{self, Write};

// fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
//     // Load the tokenizer
//     let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

//     // Load the ONNX model bytes
//     let model_bytes = fs::read("llama3_model.onnx")?;

//     // Create the environment
//     let environment = Environment::builder()
//         .with_name("llama3_inference")
//         .with_log_level(onnxruntime::LoggingLevel::Warning)
//         .build()?;

//     // Load the ONNX model from memory
//     let session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_model_from_memory(&model_bytes)?;

//     loop {
//         // Get the prompt from the user
//         print!("Enter your prompt: ");
//         io::stdout().flush()?;
//         let mut prompt = String::new();
//         io::stdin().read_line(&mut prompt)?;

//         // Tokenize the input
//         let encoding = tokenizer.encode(prompt.trim(), true)?;
//         let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
//         let input_tensor = Array::from_shape_vec((1, input_ids.len()), input_ids)?;

//         // Run the model
//         let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor.into()])?;

//         // Decode the output
//         let output_ids: Vec<i64> = outputs[0].iter().map(|&id| id as i64).collect();
//         let response = tokenizer.decode(output_ids, true)?;

//         // Print the response
//         println!("Response: {}", response);
//     }
// }

// use onnxruntime::{environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel, session::Session};
// use tokenizers::Tokenizer;
// use std::error::Error;
// use std::fs;
// use std::io::{self, Write};

// fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
//     // Load the tokenizer
//     let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

//     // Load the ONNX model bytes
//     let model_bytes = fs::read("llama3_model.onnx")?;

//     // Create the environment
//     let environment = Environment::builder()
//         .with_name("llama3_inference")
//         .with_log_level(onnxruntime::LoggingLevel::Warning)
//         .build()?;

//     // Load the ONNX model from memory
//     let session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_model_from_memory(&model_bytes)?;

//     loop {
//         // Get the prompt from the user
//         print!("Enter your prompt: ");
//         io::stdout().flush()?;
//         let mut prompt = String::new();
//         io::stdin().read_line(&mut prompt)?;

//         // Tokenize the input
//         let encoding = tokenizer.encode(prompt.trim(), true)?;
//         let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
//         let input_tensor = Array::from_shape_vec((1, input_ids.len()), input_ids)?;

//         // Run the model
//         let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor.into()])?;

//         // Decode the output
//         let output_ids: Vec<i64> = outputs[0].iter().map(|&id| id as i64).collect();
//         let response = tokenizer.decode(output_ids, true)?;

//         // Print the response
//         println!("Response: {}", response);
//     }
// }



use onnxruntime::{environment::Environment, ndarray::Array, tensor::OrtOwnedTensor, GraphOptimizationLevel,session::Session};
use tokenizers::Tokenizer;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Load the tokenizer
    let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

    // Load the ONNX model bytes
    let model_bytes = fs::read("llama3_model.onnx")?;

    // Create the environment
    let environment = Environment::builder()
        .with_name("llama3_inference")
        .with_log_level(onnxruntime::LoggingLevel::Warning)
        .build()?;

    // Load the ONNX model from memory
    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_model_from_memory(&model_bytes)?;

    loop {
        // Get the prompt from the user
        print!("Enter your prompt: ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;

        // Tokenize the input
        let encoding = tokenizer.encode(prompt.trim(), true)?;
        let input_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();
        let input_tensor = Array::from_shape_vec((1, input_ids.len()), input_ids)?;

        // Run the model
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor.into()])?;

        // Decode the output
        let output_ids: Vec<i64> = outputs[0].iter().map(|&id| id as i64).collect();
        let response = tokenizer.decode(output_ids, true)?;

        // Print the response
        println!("Response: {}", response);
    }
}
