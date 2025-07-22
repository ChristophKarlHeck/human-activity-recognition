import tensorflow as tf

# --- Define these based on YOUR .pb model ---
GRAPH_PB_PATH = 'frozen_har.pb'  # Path to your .pb file
OUTPUT_TFLITE_PATH = 'model.tflite' # Desired output .tflite file path

# Input and output tensor names from your .pb model
# You might need to inspect your .pb model (e.g., using Netron or TensorFlow's summarize_graph tool)
# to find the correct names of the input and output tensors.
# The Java code you shared used "inputs" and "y_".
input_tensor_names = ["input"] # Example, replace with your model's input tensor name(s)
output_tensor_names = ["y_"]    # Example, replace with your model's output tensor name(s)

# For some models, especially older ones, you might need to specify input shapes
# if they are not fully defined in the .pb file.
# Example: INPUT_SHAPES = {"inputs": [1, 200, 3]} # batch_size, height/timesteps, width/features

# Load the frozen graph
with tf.io.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    for op in graph.get_operations():
        print(op.name)

# Create the converter
# For TF 2.x:
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=GRAPH_PB_PATH,
    input_arrays=input_tensor_names,
    output_arrays=output_tensor_names,
    # If you needed to specify input_shapes (uncomment and adjust if necessary):
    input_shapes={"input": [1, 200, 3]} 
)

# Optional: Apply optimizations (e.g., quantization)
# converter.optimizations = [tf.lite.Optimize.DEFAULT] # For size reduction and potential speedup

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(OUTPUT_TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted successfully to {OUTPUT_TFLITE_PATH}")