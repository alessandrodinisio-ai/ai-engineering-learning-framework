# AI Engineering Glossary

## A

### Agent
- **What people say:** "An autonomous AI that thinks and acts on its own"
- **What it actually means:** A while loop: the LLM decides which tool to call next, executes it, observes the result, and repeats
- **Why it's called that:** Borrowed from philosophy — an "agent" is anything that can act in the world. In AI, it just means "LLM + tools + loop"

### Attention
- **What people say:** "How AI focuses on what matters"
- **What it actually means:** A mechanism where each token computes a weighted sum over all other tokens' values, with weights determined by how relevant they are (computed via the dot product of query and key vectors)
- **Why it's called that:** The 2017 paper "Attention Is All You Need" named it by analogy to human selective attention

### Alignment
- **What people say:** "Making AI safe"
- **What it actually means:** A technical challenge: making AI systems behave according to human intent, values, and preferences, including edge cases the designers didn't anticipate

### Autoregressive
- **What people say:** "AI generates one word at a time"
- **What it actually means:** A model that predicts the next token based on all preceding tokens, then feeds that prediction back as input for the next step. GPT, LLaMA, and Claude are all autoregressive.

### Activation Function
- **What people say:** "The nonlinear thing between layers"
- **What it actually means:** A function applied after each linear layer to introduce nonlinearity. Without it, stacking any number of linear layers collapses to a single linear transform. ReLU, GELU, and SiLU are most common. The choice directly affects whether gradients can flow during training.

### Adam (Optimizer)
- **What people say:** "The default optimizer"
- **What it actually means:** Adaptive Moment Estimation. Combines momentum (first moment) with per-parameter adaptive learning rates (second moment). Includes bias correction for early steps. Works well on most tasks with little tuning.

### AdamW
- **What people say:** "Adam but better"
- **What it actually means:** Adam with decoupled weight decay. In standard Adam, L2 regularization gets scaled by the per-parameter adaptive learning rate — not what you want. AdamW applies weight decay directly to the weights, independent of gradient statistics. The default optimizer for training transformers.

### Autograd
- **What people say:** "Automatic gradient computation"
- **What it actually means:** A system that records operations on tensors and computes gradients automatically via reverse-mode differentiation. PyTorch's autograd builds the computation graph on-the-fly (dynamic graph); JAX uses function transforms (grad). It makes backpropagation practical — you write only the forward pass and the framework computes all derivatives for you.

## B

### Batch Size
- **What people say:** "How many samples at once"
- **What it actually means:** The number of training examples processed in one forward/backward pass before updating weights. Larger batches give more stable gradient estimates but use more memory. Typical values: 32–512 for training, larger for inference. Batch size and learning rate are coupled — double the batch, double the learning rate (linear scaling rule).

### Backpropagation
- **What people say:** "How neural networks learn"
- **What it actually means:** An algorithm that applies the chain rule backward through the network to compute how much each weight contributed to the error, then adjusts weights proportionally
- **Why it's called that:** Errors propagate backward from output to input, layer by layer

## C

### Context Window
- **What people say:** "How much AI can remember"
- **What it actually means:** The maximum number of tokens (input + output) that fits in a single API call. It's not memory — it's a fixed-size buffer that resets on every call

### Chain of Thought (CoT)
- **What people say:** "Making AI think step by step"
- **What it actually means:** A prompting technique that makes the model show its reasoning steps, improving accuracy on multi-step problems because each step conditions the generation of the next token

### CNN (Convolutional Neural Network)
- **What people say:** "Image AI"
- **What it actually means:** A neural network that uses convolution operations (sliding filters over the input) to detect local patterns. Stacked convolution layers detect progressively complex features: edges, textures, objects.

### CUDA
- **What people say:** "GPU programming"
- **What it actually means:** NVIDIA's parallel computing platform. Lets you run matrix operations simultaneously on thousands of GPU cores. PyTorch and TensorFlow use CUDA under the hood.

### Chunking
- **What people say:** "Splitting documents into pieces"
- **What it actually means:** Breaking text into segments before embedding for retrieval. Chunk size determines result granularity. Too small: lose context. Too large: dilute relevance. Common strategies: fixed-length with overlap, sentence-based, or semantic segmentation. Typical chunk size: 256–512 tokens with 10–20% overlap.

### Contrastive Learning
- **What people say:** "Learning by comparison"
- **What it actually means:** Training by pulling similar pairs closer and pushing dissimilar pairs apart in embedding space. CLIP uses this: matching image-text pairs vs. non-matching ones.

### Cosine Similarity
- **What people say:** "How similar two vectors are"
- **What it actually means:** The cosine of the angle between two vectors: dot(a, b) / (||a|| * ||b||). Ranges from -1 (opposite) to 1 (same direction). Ignores magnitude, only measures direction. The standard similarity metric for embeddings and semantic search.

### Cross-Entropy
- **What people say:** "The classification loss"
- **What it actually means:** Measures the gap between two probability distributions. For classification: -sum(y_true * log(y_pred)). For language models: the negative log probability of the correct next token. Lower is better. Perplexity is just exp(cross-entropy).

## D

### Data Augmentation
- **What people say:** "Making more training data"
- **What it actually means:** Creating modified copies of existing data (rotating images, adding noise, rephrasing text) to increase training set diversity without collecting new data. Reduces overfitting.

### Decoder
- **What people say:** "The output part"
- **What it actually means:** In a transformer, the decoder uses causal (masked) self-attention so each position can only attend to earlier positions. GPT is decoder-only. BERT is encoder-only. T5 is encoder-decoder.

### Diffusion Model
- **What people say:** "AI that generates images from noise"
- **What it actually means:** A model trained to reverse a gradual noising process — it learns to predict and remove noise, generating by starting from pure noise and iteratively denoising

### DPO (Direct Preference Optimization)
- **What people say:** "Simpler RLHF"
- **What it actually means:** A training method that skips the reward model entirely — directly optimizes the language model to prefer the better response in pairs of human preferences

### Dropout
- **What people say:** "Randomly turning off neurons"
- **What it actually means:** Randomly zeroing out a fraction of activations during training. Forces the network not to rely on any single neuron. Turned off at inference. Simple but effective regularization.

## E

### Eigenvalue
- **What people say:** "That math concept from PCA"
- **What it actually means:** For a matrix A, eigenvalue lambda satisfies Av = lambda*v (where v is some vector). It tells you how much the matrix scales a vector in that direction. Large eigenvalue = high-variance direction in data.

### Embedding
- **What people say:** "Some AI magic that turns words into numbers"
- **What it actually means:** A learned mapping from discrete items (words, images, users) to dense vectors in a continuous space, where similar items end up close together
- **Why it's called that:** The items are "embedded" in a geometric space where distance has meaning

### Encoder
- **What people say:** "The input part"
- **What it actually means:** In a transformer, the encoder uses bidirectional self-attention so each position can attend to all positions. BERT is encoder-only. Good at understanding tasks (classification, NER) but not generation.

### Epoch
- **What people say:** "One pass through the data"
- **What it actually means:** Exactly what it says. One complete pass through every sample in the training set. Multiple epochs = seeing the data multiple times. More epochs may improve learning but risk overfitting.

## F

### Feature
- **What people say:** "A column in the data"
- **What it actually means:** A measurable property of the data. In classical ML, you hand-engineer features. In deep learning, the network learns features automatically from raw data.

### Few-Shot
- **What people say:** "Giving AI a few examples first"
- **What it actually means:** Including a small number of input-output examples in the prompt before the actual task. Usually 3–5. The model pattern-matches on these to figure out the format and behavior you want. Compare to zero-shot (no examples) and fine-tuning (thousands of examples baked into weights).

### Fine-tuning
- **What people say:** "Training AI on your data"
- **What it actually means:** Starting from a pre-trained model's weights and continuing training on a smaller, task-specific dataset. Only updates existing weights, doesn't add new knowledge from scratch

### Function Calling
- **What people say:** "AI that can use tools"
- **What it actually means:** A structured way for LLMs to request execution of external functions. You describe tools with JSON Schema definitions, the model outputs a structured JSON object specifying which function to call and what arguments to pass, your code executes it, and the result goes back to the model. Not the same as an agent — function calling is the mechanism, the agent is the loop.

## G

### Guardrails
- **What people say:** "Safety filters for AI"
- **What it actually means:** Input/output validation layers around an LLM that detect and block harmful content, prompt injection attempts, PII leaks, or off-topic responses. Typically a pipeline: input filter → LLM → output filter. Can be rule-driven (regex, keyword lists) or model-driven (a classifier scoring safety).

### GPT
- **What people say:** "ChatGPT" or "the AI"
- **What it actually means:** Generative Pre-trained Transformer — a specific architecture using a decoder-only transformer trained on massive text corpora to predict the next token
- **Why it's called that:** Generative (produces text), Pre-trained (trained once on large data, then adapted), Transformer (the architecture)

### GAN (Generative Adversarial Network)
- **What people say:** "Two AIs fighting each other"
- **What it actually means:** A generator network tries to produce realistic data while a discriminator network tries to tell real from fake. They train together: the generator gets better at fooling the discriminator, and the discriminator gets better at spotting fakes.

### Gradient
- **What people say:** "The slope"
- **What it actually means:** A vector of partial derivatives pointing in the direction of steepest ascent. In ML, you move in the opposite direction (gradient descent) to minimize loss.

### Gradient Descent
- **What people say:** "How AI improves"
- **What it actually means:** An optimization algorithm that adjusts parameters in the direction that most steeply decreases the loss function — like walking downhill in a high-dimensional landscape

## H

### Hyperparameter
- **What people say:** "Settings you tune"
- **What it actually means:** Values set before training that control the training process itself: learning rate, batch size, number of layers, dropout rate. Unlike model parameters (weights), these are not learned from data.

### Hallucination
- **What people say:** "AI is lying" or "making things up"
- **What it actually means:** The model generates text that sounds plausible but is grounded in neither its training data nor the given context — it's completing patterns, not retrieving facts

## I

### Inference
- **What people say:** "Running AI"
- **What it actually means:** Using a trained model to make predictions on new data. No weight updates happen. This is what you do in production: send input, get output.

### Inductive Bias
- **What people say:** Never heard of it
- **What it actually means:** Assumptions baked into the model architecture. CNNs assume local patterns matter (convolution). RNNs assume order matters (sequential processing). Transformers assume everything might relate to everything (attention). The right bias helps the model learn faster with less data.

### JAX
- **What people say:** "Google's ML framework"
- **What it actually means:** A NumPy-compatible library with automatic differentiation (grad), JIT compilation (jit), automatic vectorization (vmap), and multi-device parallelism (pmap). Unlike PyTorch's object-oriented style, JAX is purely functional — no hidden state, no in-place mutations. Google DeepMind uses it for AlphaFold, Gemini, and large-scale research.

## K

### KV Cache
- **What people say:** "Making inference faster"
- **What it actually means:** Caching the key and value matrices from previous tokens during autoregressive generation so you don't recompute them at each step. Trading memory for speed. Critical for fast LLM inference.

## L

### Latent Space
- **What people say:** "The hidden representation"
- **What it actually means:** A compressed, learned representation space where similar inputs map to nearby points. Autoencoders, VAEs, and diffusion models all operate in latent space. It's lower-dimensional than the input but captures important structure.

### Learning Rate
- **What people say:** "How fast AI learns"
- **What it actually means:** A scalar controlling step size during gradient descent. Too high: overshoot the minimum and diverge. Too low: converge too slowly or get stuck. The single most important hyperparameter.

### LLM (Large Language Model)
- **What people say:** "AI" or "the brain"
- **What it actually means:** A transformer-based neural network trained to predict the next token in a sequence, with billions of parameters, trained on internet-scale text data

### LoRA (Low-Rank Adaptation)
- **What people say:** "Efficient fine-tuning"
- **What it actually means:** Instead of updating all weights, insert small low-rank matrices alongside the original weights. Train only these small matrices, reducing memory by 10–100×

### Loss Function
- **What people say:** "How wrong AI is"
- **What it actually means:** A function measuring the gap between predicted and actual output. Training minimizes this function. MSE for regression, cross-entropy for classification, contrastive loss for embeddings. The choice of loss defines what "good" means for the model.

## M

### Mixed Precision
- **What people say:** "Training speedup trick"
- **What it actually means:** Using float16 for forward pass and most operations (faster, less memory) but keeping gradient accumulation and weight updates in float32 (more precise). Gets ~2× speedup with negligible accuracy loss.

### MoE (Mixture of Experts)
- **What people say:** "Model only runs part of itself"
- **What it actually means:** A model with many "expert" subnetworks where a routing mechanism sends each input to only a few experts. The full model is large but each forward pass is cheap because most experts are skipped. Mixtral and GPT-4 use this.

### MCP (Model Context Protocol)
- **What people say:** "A way for AI to use tools"
- **What it actually means:** An open protocol (JSON-RPC over stdio/HTTP) standardizing how AI applications connect to external data sources and tools, with typed schemas for tools, resources, and prompts

## N

### NaN (Not a Number)
- **What people say:** "Training crashed"
- **What it actually means:** A floating-point value representing an undefined result (0/0, inf-inf). NaN loss during training usually means: learning rate too high, exploding gradients, log of zero, or division by zero. The first thing to check when training fails.

### Normalization
- **What people say:** "Scaling data"
- **What it actually means:** Adjusting values to a standard range. Batch normalization normalizes within a batch. Layer normalization normalizes across features. Both stabilize training and allow higher learning rates.

## O

### Overfitting
- **What people say:** "Model memorized the data"
- **What it actually means:** The model performs well on training data but poorly on unseen data. It learned noise, not signal. Fixes: more data, regularization (dropout, weight decay), early stopping, data augmentation, simpler model.

### Optimizer
- **What people say:** "The thing that updates weights"
- **What it actually means:** An algorithm that uses gradients to update model parameters. SGD is the simplest. Adam is the most common. Each optimizer has different properties: convergence speed, memory usage, sensitivity to hyperparameters.

## P

### Parameter
- **What people say:** "Model size"
- **What it actually means:** A learnable value in the model's parameter matrices, typically a weight or bias. "7B parameters" means 7 billion learnable numbers. Each float32 parameter takes 4 bytes, so 7B parameters = 28GB just for weights.

### Perplexity
- **What people say:** "How confused the model is"
- **What it actually means:** The exponential of average cross-entropy loss. Lower is better. Perplexity of 10 means the model's uncertainty is equivalent to randomly guessing among 10 tokens at each step.

### Precision & Recall
- **What people say:** "Accuracy metrics"
- **What it actually means:** Precision = of the items you flagged, how many were correct. Recall = of all correct items, how many did you find. They trade off: catching every spam email (high recall) means more false positives (low precision). F1 score is their harmonic mean. Look at precision when false positives are costly, recall when misses are costly.

### Prompt Engineering
- **What people say:** "Talking to AI the right way"
- **What it actually means:** Designing input text to reliably produce desired outputs — including system prompts, few-shot examples, format instructions, and chain-of-thought triggers

### Prompt Injection
- **What people say:** "Hacking AI with text"
- **What it actually means:** An attack where malicious text in the input overrides the system prompt or instructions. Direct injection: user types "ignore previous instructions." Indirect injection: retrieved documents contain hidden instructions. The LLM equivalent of SQL injection. No complete fix — defense relies on layered input validation, output filtering, and privilege isolation.

## Q

### QLoRA
- **What people say:** "Even cheaper LoRA"
- **What it actually means:** Quantized LoRA. Keeps frozen base model weights in 4-bit precision (NF4 format) while training LoRA adapters in 16-bit. Saves 3–4× more memory than standard LoRA. A 7B model needing 14GB with LoRA fits in 4–6GB with QLoRA. Quality within 1% of full fine-tuning on most benchmarks.

## R

### RAG (Retrieval-Augmented Generation)
- **What people say:** "AI that can search"
- **What it actually means:** A pattern where you retrieve relevant documents from a knowledge base (using embedding similarity), stuff them into the prompt, and have the LLM answer based on that context
- **Why it's called that:** Retrieval (find documents) + Augmented (add to prompt) + Generation (LLM writes the answer)

### RLHF (Reinforcement Learning from Human Feedback)
- **What people say:** "How they made AI useful"
- **What it actually means:** A training pipeline: (1) collect human preferences on model outputs, (2) train a reward model on those preferences, (3) optimize the LLM with PPO to produce higher-reward outputs

### Quantization
- **What people say:** "Making the model smaller"
- **What it actually means:** Reducing model weight precision from float32 (4 bytes) to int8 (1 byte) or int4 (0.5 bytes). Trades a small amount of accuracy for 4–8× memory savings and faster inference. GPTQ, AWQ, and GGUF are common formats.

### ReLU
- **What people say:** "The activation function"
- **What it actually means:** Rectified Linear Unit: f(x) = max(0, x). The simplest nonlinear activation. Fast to compute, doesn't saturate for positive values. Used everywhere because it works and is cheap. Variants: LeakyReLU, GELU, SiLU.

### ROUGE
- **What people say:** "Summarization metric"
- **What it actually means:** Recall-Oriented Understudy for Gisting Evaluation. Measures overlap between generated and reference text. ROUGE-1 counts unigram hits, ROUGE-2 counts bigram hits, ROUGE-L finds the longest common subsequence. Cheap to compute but only measures surface similarity — two sentences with the same meaning but different words score poorly.

## S

### Semantic Search
- **What people say:** "Smart search that understands meaning"
- **What it actually means:** Finding documents by meaning rather than keyword matching. Embed the query and all documents into the same vector space, then return documents whose embeddings are closest to the query embedding. "Payment failed" finds "transaction declined" even though they share no words. Powered by embedding models + vector databases.

### Streaming
- **What people say:** "Watching the answer appear word by word"
- **What it actually means:** The LLM sends tokens as they're generated rather than waiting for the full response. Uses Server-Sent Events (SSE) or WebSocket protocols. Reduces perceived latency for the first token from seconds to milliseconds. Essential for production chat interfaces. Each chunk contains a delta (partial token or word).

### Self-Attention
- **What people say:** "How the model decides what to focus on"
- **What it actually means:** Each token computes query, key, and value vectors. The attention weight between two tokens = the dot product of their query and key, scaled and passed through softmax. Output = weighted sum of value vectors. Lets every token see every other token.

### SFT (Supervised Fine-Tuning)
- **What people say:** "Teaching the model to follow instructions"
- **What it actually means:** Fine-tuning a pre-trained model on (instruction, response) pairs. The model learns to generate responses given instructions. This is how a base model becomes a chat model.

### Softmax
- **What people say:** "Turning numbers into probabilities"
- **What it actually means:** softmax(x_i) = exp(x_i) / sum(exp(x_j)). Converts an arbitrary vector of real numbers into a probability distribution (all positive, sums to 1). Used in classification heads, attention weights, and anywhere you need probabilities.

### Swarm
- **What people say:** "A group of AI agents working together like bees"
- **What it actually means:** Multiple agents sharing state and coordinating via message passing, with emergent behavior arising from simple individual rules rather than central control

## T

### System Prompt
- **What people say:** "AI's instructions"
- **What it actually means:** A special message at the start of a conversation that sets the model's behavior, persona, and constraints. Processed before user messages. Invisible to users in most UIs. Defines what the model should do, shouldn't do, tone, format preferences, and domain focus. Unlike user prompts — system prompts are set by developers.

### Tensor
- **What people say:** "A multi-dimensional array"
- **What it actually means:** The fundamental data structure in deep learning frameworks. 0D tensor is a scalar, 1D is a vector, 2D is a matrix, 3D+ is a tensor. In PyTorch and JAX, tensors track their computation history for automatic differentiation and can reside on CPU or GPU. All inputs, outputs, weights, and gradients in neural networks are tensors.

### Token
- **What people say:** "A word"
- **What it actually means:** A subword unit produced by a tokenizer (like BPE), typically 3–4 characters in English. "unbelievable" might be 3 tokens: "un" + "believ" + "able"

### Temperature
- **What people say:** "The creativity setting"
- **What it actually means:** A scalar that divides logits before softmax. Temperature=1 is default. Higher = flatter distribution = more random output. Lower = sharper distribution = more deterministic. Temperature=0 is argmax (always pick the most probable token).

### Transfer Learning
- **What people say:** "Using a pre-trained model"
- **What it actually means:** Taking a model trained on one task and adapting it to another. Early layers learn general features (edges, syntactic patterns) that transfer. Only later layers need task-specific training. This is why you can fine-tune BERT for any NLP task.

### Transformer
- **What people say:** "The architecture behind modern AI"
- **What it actually means:** A neural network architecture that uses self-attention (letting each position attend to every other position) instead of recurrence to process sequences, enabling massive parallelization
- **Why it's called that:** It transforms input representations into output representations through attention layers

## U

### Underfitting
- **What people say:** "Model isn't learning"
- **What it actually means:** The model is too simple to capture patterns in the data. Training loss stays high. Fixes: more parameters, more layers, longer training, less regularization, better features.

## V

### VAE (Variational Autoencoder)
- **What people say:** "A type of generative model"
- **What it actually means:** An autoencoder that learns a smooth latent space by forcing the encoder output to follow a Gaussian distribution. You can sample from this distribution and decode to generate new data. The reparameterization trick makes it trainable via backpropagation.

### Vector Database
- **What people say:** "A database for AI"
- **What it actually means:** A database optimized for storing vectors (dense float arrays) and performing fast approximate nearest-neighbor search. The core operation in similarity search, RAG, and recommendation systems.

## W

### Weight
- **What people say:** "What the model learned"
- **What it actually means:** A single number in the model's parameter matrices. A linear layer with input size 768 and output size 3072 has 768×3072 = 2,359,296 weights. Training adjusts each weight to minimize the loss function.

### Weight Decay
- **What people say:** "Regularization"
- **What it actually means:** Adding a penalty to the loss function proportional to the magnitude of the weights. Equivalent to L2 regularization. Prevents weights from growing too large. Typical values: 0.01–0.1.

## Z

### Zero-Shot
- **What people say:** "No training needed"
- **What it actually means:** Using a model on a task it wasn't explicitly trained for, with no task-specific examples in the prompt. The model relies on pre-training to generalize. Works because large models have seen enough variety to handle new task formats.
