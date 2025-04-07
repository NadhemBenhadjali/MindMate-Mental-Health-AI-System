# MindMate: Fine-Tuning and Testing
# Mindmate Website Repository

This repository contains the source code for the **Mindmate Website**.
📂 **Repository Link:** [AY00Z/EMBS-Challenge](https://github.com/AY00Z/EMBS-Challenge)
## Introduction to LLaMA 3-8B Chat
The **LLaMA 3-8B Chat** model is a powerful, open-weight large language model (LLM) developed to provide exceptional conversational AI capabilities. With 8 billion parameters, it strikes a balance between performance and resource efficiency, making it ideal for fine-tuning and deployment across various applications.

### Why LLaMA 3-8B Chat is Powerful:
1. **Scalability**: With its 8 billion parameters, LLaMA 3-8B Chat delivers high-quality outputs while remaining computationally efficient compared to larger models.
2. **Adaptability**: The model can be fine-tuned to specific domains, such as mental health conversations, customer support, or technical troubleshooting, making it highly versatile.
3. **Open-Weight Access**: Being open-source allows developers and researchers to customize and optimize the model without restrictions.
4. **Performance**: LLaMA 3-8B Chat exhibits state-of-the-art performance on various NLP benchmarks, enabling tasks such as text generation, summarization, and extraction with high accuracy.
5. **Resource-Efficient**: Its size ensures that it can be deployed on consumer-grade GPUs and cloud platforms without requiring significant resources.

---

## Notebook Descriptions

### 1. `llama-3-8b-chat-hf-finetuning.ipynb`
This notebook covers the fine-tuning process of the **LLaMA 3-8B Chat** model using Hugging Face's Transformers library. It demonstrates how to:
- Load and combine 3 mental health conversation datasets:
  - `Amod/mental_health_counseling_conversations`
  - `mpingale/mental-health-chat-dataset`
  - `heliosbrahma/mental_health_chatbot_dataset`
- Configure training parameters
- Fine-tune the model using PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation)

**Training Configuration:**
- **Training Time**: Approximately 3 hours on a **Kaggle GPU P100**
- **Key Parameters**:
   - Learning rate: `6e-5`
   - Batch size: `2` (train), `1` (eval)
   - Epochs: `2`
   - Optimizer: `paged_adamw_32bit`
   - Mixed Precision: `fp16=True`, `bf16=False`
   - LoRA Configuration: `lora_alpha=16`, `r=8`, `lora_dropout=0.1`
   - Scheduler: Constant learning rate
**Training Observations**

The model was fine-tuned using an NVIDIA P100 GPU on Kaggle for a duration of three hours. Training loss improved significantly over time, indicating effective learning. Key observations include:

- **Initial loss:** ~2.66  
- **Final loss:** ~0.87  
- The loss steadily decreased, with notable improvement after ~700 steps and stabilization after ~4000 steps.

**Training Steps and Loss**

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


**Fine-tuned weights** and the notebook can be **directly loaded from Kaggle**.

### 2. `mindmate-testing-and-prompt-engineering.ipynb`
This notebook is focused on:
- Testing the fine-tuned LLaMA 3-8B Chat model
- Performing prompt engineering to optimize outputs for mental health conversation extraction tasks

It includes detailed testing with sample conversations and prompt strategies.

---

## Prompt Engineering Examples

### MindMate Psychologist
**Prompt**: You are a psychologist assisting a patient. Please respond empathetically, actively listen, and encourage self-reflection. Avoid offering quick fixes or medical advice. Ask open-ended questions to explore the patient’s thoughts and feelings. Use a compassionate tone in your responses.

---

### MindMate Analyser
**Prompt**: Analyze the conversation below to extract the symptoms mentioned by the patient, providing specific evidence from the dialogue to support each identified symptom. For each symptom, include the exact statements or phrases that indicate its presence. Conclude your analysis with a summary of the patient's condition based on the detected symptoms.

---

## Quick Access on Kaggle
Both notebooks and the fine-tuned model weights are available on Kaggle for direct use:
- [LLaMA 3-8B Fine-Tuning Notebook](https://www.kaggle.com/code/nadhembenhadjali/llama-3-8b-chat-hf-finetuning)
- [Fine-tuned Model Weights](https://www.kaggle.com/models/nadhembenhadjali/mindmate)
- [mindmate-testing-and-prompt-engineering.ipynb](https://www.kaggle.com/code/nadhembenhadjali/mindmate-testing-and-prompt-engineering)

To use them directly:
1. Go to Kaggle and open the notebook.
2. Load the fine-tuned weights using Kaggle's file system.
3. Follow the usage instructions in the notebook to test and analyze the model outputs.

Replace the links above with the appropriate Kaggle URLs where your notebooks and model weights are hosted.

---

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/NadhemBenhadjali/your-repository-name.git
   ```
2. Open the notebooks using Kaggle.
3. To load the fine-tuned weights, download them directly from Kaggle and integrate them into the testing notebook.

---

## Dependencies
- Python 3.8+
- Transformers (Hugging Face)
- PyTorch
- PEFT (LoRA)
- Kaggle API (optional for direct downloads)

---

## License
This project is licensed under the MIT License.

---

## Author
Nadhem Benhadjali

For any questions, feel free to open an issue or contact me.
