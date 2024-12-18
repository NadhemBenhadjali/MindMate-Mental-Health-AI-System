# Model Card for Mental Health Fine-Tuned LLM

## Model Details

### Model Description

This model is a fine-tuned version of a large language model, specifically adapted for applications in mental health and psychotherapy. Leveraging the PEFT (Parameter-Efficient Fine-Tuning) library, this model is designed to assist in text generation tasks, such as answering user queries, providing suggestions, and generating therapeutic responses. It was fine-tuned using datasets relevant to mental health conversations to enhance its applicability in this domain.

- **Language(s):** English  
- **License:** MIT  
- **Fine-tuned from model:** Llama 3 8B Chat

---

## Uses

### Direct Use

This model is intended for:

- Assisting mental health professionals with preliminary insights based on user queries.
- Providing information and psychoeducational support to users seeking guidance.
- Serving as a chatbot for mental health applications to simulate supportive conversations.

### Downstream Use

- Fine-tuning for specific mental health sub-domains or disorders.
- Integration into digital therapeutics or self-help platforms.

### Out-of-Scope Use

- Providing clinical diagnosis or treatment without supervision by a qualified mental health professional.
- Use cases that may involve malicious manipulation or exploitation of vulnerable users.

---

## Bias, Risks, and Limitations

### Risks

- Potential for reinforcing biases present in the training data.

### Recommendations

- Always have responses reviewed by a licensed mental health professional before deployment.
- Use the model as a supplementary tool, not as a replacement for human expertise.

---

## How to Get Started with the Model

Use the following code snippet to begin using the model:

```python
from transformers import pipeline

# Load the fine-tuned model
generator = pipeline("text-generation", model="/kaggle/input/llama-3/transformers/8b-chat-hf/1")

# Example usage
response = generator("I feel like I don't exist and my body is not my own.")
print(response)
```

---

## Training Details

### Training Data

This model was fine-tuned on:

- [mpingale/mental-health-chat-dataset](https://huggingface.co/datasets/mpingale/mental-health-chat-dataset)
- [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- [heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset)

### Training Procedure

#### Preprocessing

- Text cleaning to remove irrelevant or noisy data.
- Tokenization and preparation for the Llama architecture.

#### Training Hyperparameters

- **Learning rate:** 5e-5
- **Batch size:** 16
- **Epochs:** 3
- **Optimizer:** AdamW
- **Mixed Precision:** FP16

#### Training Observations

The model was fine-tuned using an NVIDIA P100 GPU on Kaggle for a duration of three hours. Training loss improved significantly over time, indicating effective learning. Key observations include:

- **Initial loss:** ~2.66  
- **Final loss:** ~0.87  
- The loss steadily decreased, with notable improvement after ~700 steps and stabilization after ~4000 steps.

#### Training Steps and Loss

| Step | Training Loss |
|------|---------------|
| 100  | 2.664700      |
| 200  | 2.395100      |
| 300  | 2.333700      |
| 400  | 2.139800      |
| 500  | 2.075500      |
| 600  | 2.002000      |
| 700  | 1.966200      |
| 800  | 1.922300      |
| 900  | 1.977200      |
| ...  | ...           |
| 6200 | 0.868600      |

This progression suggests that the model is close to convergence, with steady improvements over the training period.

---

## Environmental Impact

Carbon emissions can be estimated using the Machine Learning Impact calculator.

- **Hardware Type:** NVIDIA A100 GPUs
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

---

## Technical Specifications

### Model Architecture and Objective

- **Base Model:** Llama 3 8B Chat  
- **Objective:** Causal Language Modeling

### Compute Infrastructure

- **Hardware:** NVIDIA A100 GPUs  
- **Software:** PyTorch, PEFT 0.13.2

---

## Model Card Contact

For questions or feedback, please contact [Your Contact Information].

---

## Framework Versions

- **PEFT:** 0.13.2
