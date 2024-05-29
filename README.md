# Llama3 ONNX Inference with Rust

This project demonstrates how to convert a Llama3 model to ONNX format using Python and run inference using Rust. The steps include converting the model and tokenizer to ONNX and using Rust to interact with the model in the terminal.

## Prerequisites

- Python 3.x , mine is 3.11
- Rust and Cargo
- Pip package manager
- Openssl 

## Step 1: Convert Llama3 Model to ONNX using pytorch

### Install Python Dependencies

First, ensure you have the required Python packages installed in a  torch environment:

```bash
pip install transformers torch onnx onnxruntime
```

### Convert the Model

Create a script `convert_to_onnx.py` with the following content to convert the Llama3 model to ONNX format:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Llama3 model and tokenizer from Hugging Face
model_name = "model-name"  # Replace with the actual model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dummy input for tracing the model
dummy_input = tokenizer("Translate English to French: Hello, how are you?", return_tensors="pt")

# Export the model to ONNX format
torch.onnx.export(
    model, 
    (dummy_input["input_ids"],), 
    "llama3_model.onnx",
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size", 1: "sequence_length"}},
    opset_version=11
)
```

Run the script to generate the `llama3_model.onnx` file:

```bash
python convert_to_onnx.py
```

or

To accomplish your goal of converting the Llama3 model to ONNX format and running it via terminal prompts in Rust, you'll need to follow a series of steps. Here's a high-level overview of how you might approach this project:

### 1. Obtain the Llama3 Model

First, you need to obtain the Llama3 model. This model might be available directly from the developers or through a model repository. Ensure you have the right to use it and that it fits your requirements.

### 2. Convert Llama3 to ONNX Format

The conversion of a model to ONNX format depends on the framework it was originally trained in. Most modern frameworks like PyTorch or TensorFlow have tools to help with this conversion:

- **TensorFlow to ONNX:**

  ```bash
  pip install tf2onnx
  python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
  ```

- **PyTorch to ONNX:**

  ```python
  import torch
  import torch.onnx

  # Load your pre-trained model
  model = YourModel()
  model.load_state_dict(torch.load('model_weights.pth'))

  # Set the model to inference mode
  model.eval()

  # An example input you would normally provide to your model's forward() method.
  example_input = torch.rand(1, 3, 224, 224)  # Adjust the shape according to your model

  # Export the model
  torch.onnx.export(model, example_input, "model.onnx", export_params=True, opset_version=10)
  ```

### 3. Run ONNX Model in Rust

To run the ONNX model in Rust, you can use the `onnxruntime` crate which provides bindings to the ONNX Runtime.

- **Add onnxruntime to Cargo.toml:**

  ```toml
  [dependencies]
  onnxruntime = "0.1.3"
  ```

- **Write Rust Code to Load and Run Model:**

  Here's a basic example to get you started:

  ```rust
  use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel, Session};

  fn main() -> onnxruntime::Result<()> {
      let environment = Environment::builder()
          .with_name("test")
          .build()?;
      let session = environment.new_session_builder()?
          .with_optimization_level(GraphOptimizationLevel::Basic)?
          .with_model_from_file("model.onnx")?;

      // Prepare input. For demonstration, let's assume the model expects a float tensor.
      let input_shape = vec![1, 3, 224, 224]; // Adjust the shape according to your model
      let input_tensor_values = vec![0.0f32; 1 * 3 * 224 * 224]; // Fill with zeros or your actual input data
      let input_tensor = vec![input_tensor_values.into()];

      // Run the session
      let outputs: Vec<OrtOwnedTensor<f32>> = session.run(input_tensor)?;

      // Handle outputs
      println!("Model output: {:?}", outputs[0]);

      Ok(())
  }
  ```

### 4. Integrating Terminal Input

You can extend the above Rust code to take input from the terminal using `std::io` to make it interactive:

```rust
use std::io;

fn main() -> onnxruntime::Result<()> {
    println!("Enter your input: ");
    let mut input_text = String::new();
    io::stdin().read_line(&mut input_text).expect("Failed to read line");

    // Convert `input_text` to the input tensor format expected by the model
    // Proceed with model prediction as shown above

    Ok(())
}
```

### Next Steps, continuing from python convert_to_onnx.py in line 53

- Ensure the model's input and output formats in the Rust code match what Llama3 expects and produces.
- Test the entire flow from converting the model to running it in Rust to handle any potential issues with data types or model operations not supported by ONNX.

This setup will allow you to work with Llama3 in an efficient and performant manner in a Rust environment.

## Step 2: Run Inference Using Rust

### Set Up Rust Project

Create a new Rust project:

```bash
cargo new llama3_inference
cd llama3_inference
```

### Add Dependencies

Add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
onnxruntime = "0.13.0"  # Check for the latest version
tokenizers = "0.13.0"
```

### Write Inference Code

Create a `src/main.rs` file with the following content:

```rust
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel};
use tokenizers::Tokenizer;
use std::error::Error;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn Error>> {
    // Load the tokenizer
    let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

    // Load the ONNX model
    let environment = Environment::builder()
        .with_name("llama3_inference")
        .with_log_level(onnxruntime::LoggingLevel::Warning)
        .build()?;
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_model_from_file("llama3_model.onnx")?;

    loop {
        // Get the prompt from the user
        print!("Enter your prompt: ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;

        // Tokenize the input
        let encoding = tokenizer.encode(prompt.trim(), true)?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let input_tensor = vec![input_ids];

        // Run the model
        let outputs: Vec<OrtOwnedTensor<f32>> = session.run(vec![input_tensor.into()])?;

        // Decode the output
        let output_ids: Vec<i64> = outputs[0].iter().map(|&id| id as i64).collect();
        let response = tokenizer.decode(output_ids, true)?;

        // Print the response
        println!("Response: {}", response);
    }
}
```

### Run the Rust Program

```bash
cargo run
```

This Rust program will prompt you for input, run the input through the ONNX model, and print the generated response. Ensure the model and tokenizer paths are correct and match the names used during conversion.

## License

This project is licensed under the MIT License.

```bash

Replace `"path/to/tokenizer.json"` and `"model-name"` with the actual paths and names of your tokenizer JSON file and Llama3 model, respectively. 
```

The latest dependencies could be looked up here at [crates](https://docs.rs/crate)
