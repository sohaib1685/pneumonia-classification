# Task 1: Pneumonia Classification using CNN with Explainability

This repository contains the end-to-end deep learning pipeline for **Task 1** of the Postdoctoral Technical Challenge. It utilizes a custom-modified ResNet-18 architecture to classify heavily downsampled ($28\times28$) chest X-ray images from the PneumoniaMNIST dataset into *Normal* or *Pneumonia* categories.

The implementation emphasizes experimental rigor, featuring a complete data pipeline, learning rate scheduling, comprehensive evaluation metrics, and visual explainability using Grad-CAM.

## 📊 Results Summary
The model achieves state-of-the-art baseline performance on the test set, demonstrating strong discriminative power suitable for clinical triage:
* **Accuracy:** 0.9247
* **Precision:** 0.9455
* **Recall:** 0.9333
* **F1-Score:** 0.9394
* **AUC:** 0.9739

*(A complete analysis of the model architecture, training dynamics, and failure cases can be found in `reports/task1_classification_report.md`)*.

---


## ⚙️ Installation & Setup

This task is designed to be fully completable on standard hardware (CPU compatible).


python -m venv env
.\env\Scripts\activate

pip install -r requirements.txt


Markdown
## ⚙️ Installation & Setup

This task is designed to be fully completable on standard hardware (CPU compatible).

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create and activate a virtual environment:

Windows:

Bash
python -m venv env
.\env\Scripts\activate
macOS/Linux:

Bash
python3 -m venv env
source env/bin/activate
3. Install the dependencies:

Bash
pip install -r requirements.txt
🚀 Usage Instructions
Training the Model
To train the model from scratch, run train.py. The script uses argparse to allow for configurable hyperparameters. By default, it runs for 10 epochs with a batch size of 64.

Bash
python task1/train.py
To run with custom hyperparameters:

Bash
python task1/train.py --epochs 15 --batch_size 32 --lr 0.005
Outputs: Training loss curves will be saved to reports/figures/, and the model weights will be saved to models/task1_model.pth.

Evaluating the Model
To generate the metrics, confusion matrix, ROC curve, and extract failure cases based on the saved weights, run:

Bash
python task1/evaluate.py
Outputs: All classification metrics will print to the console. Visualizations and sample failure cases will automatically save to the reports/figures/ directory.


## 📝 Task 2: Multimodal Medical Report Generation (VLM Integration)

In this phase, the system evolves from a binary classifier into a multimodal diagnostic assistant. By integrating **MedGemma-4B**, the project can now generate structured, natural language radiology reports that provide clinical justification for the predictions made by the Task 1 model.

### 🛠️ Technical Implementation & Model Fusion
* **Model Selection:** Utilized `google/medgemma-4b-it`, chosen for its **SigLIP vision encoder** which excels at global textural reasoning—a critical requirement for the low-resolution ($28 \times 28$) MedMNIST distribution where fine anatomical edges are absent.
* **Cross-Task Weight Integration:** * The pipeline explicitly imports the **`task1_model.pth`** weights. 
    * It uses a custom loader that strips the classification head to verify feature maps during the generation process. 
    * This allows for **Synchronous Inference**: the CNN provides a rigorous mathematical probability while the VLM provides the descriptive narrative.
* **4-Bit NF4 Quantization:** To overcome the T4 GPU's 16GB VRAM limit, we implemented **4-bit NormalFloat (NF4)** quantization using `bitsandbytes`. This optimization reduced the VLM’s memory footprint from ~9GB to ~3.5GB, enabling the simultaneous residency of the VLM, the CNN, and the image processing buffers in VRAM.



### 📦 Task-Specific Requirements
To replicate this comparative environment, the following dependencies are required:
```text
# Core Frameworks
torch>=2.0.0
torchvision
medmnist

# Multimodal & Quantization
transformers>=4.40.0    # For MedGemma/Gemma-2 support
accelerate>=0.26.0      # For device-map optimization
bitsandbytes>=0.41.1    # For 4-bit memory efficiency
huggingface_hub         # For gated model authenticationnted **4-bit NF4 Quantization** via `bitsandbytes` to fit the 4B parameter model on a single T4 GPU.

* **Prompt Engineering:** Tested Zero-Shot, Clinical Role, and Context-Aware prompting strategies to mitigate hallucinations caused by the low-resolution ($28 \times 28$) MedMNIST inputs.

📂 Weights & Hardware AdaptationLoading Logic: The script uses torch.load(MODEL_PATH, map_location=device) to ensure cross-platform compatibility. This allows weights trained on a local CPU/GPU (Windows) to be seamlessly mapped to the Colab Linux environment.Memory Management: Implemented PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" and explicit gc.collect() calls between model loading stages to prevent CUDA fragmentation.📊 

📂Comparative Findings: CNN vs. VLMAccuracy Paradox: The Task 1 ResNet-18 maintained higher diagnostic accuracy because it was trained natively on downsampled data. The VLM, pre-trained on high-definition scans, showed a Conservative Bias—often labeling subtle pneumonia as "Normal" due to a lack of high-frequency visual tokens.Hallucination of Priors: We documented the "Prior Effect," where the VLM occasionally reported on anatomical structures (like heart size or rib symmetry) that are mathematically invisible at $28 \times 28$, showing how LLMs rely on their medical training memory when visual data is ambiguous.

[!IMPORTANT]Configuration: The script automatically looks for task1_model.pth in the root directory. Ensure Task 1 is completed and weights are exported before initializing the VLM pipeline.

> [!TIP]
> **Detailed Analysis:** For the full qualitative study, sample reports, and side-by-side model comparisons, see [reports/task2_report_generation.md](./reports/task2_report_generation.md).




## 📝 Task 3:  Semantic Image Retrieval & Embedding Analysis

* **Model Dependency:** This task explicitly utilizes **`task1_model.pth`** to initialize the feature extractor.
* **Mechanism:** The system converts the ResNet-18 backbone into a 512-D embedding engine by stripping the final classification head.
* **Search Engine:** Powered by **FAISS**, providing sub-millisecond similarity retrieval based on L2 distance gradients learned during Task 1 training.

## 1. Methodology: Feature Extraction & Vector Search
The objective was to transform the diagnostic model into a searchable knowledge base using **Content-Based Image Retrieval (CBIR)**.

* **Feature Extraction:** I utilized the Task 1 ResNet-18 model as a backbone. By replacing the final classification layer with an `Identity` layer, I extracted **512-dimensional embeddings** from the Global Average Pooling layer.
* **Vector Indexing:** All 624 images from the PneumoniaMNIST test set were indexed using **FAISS (Facebook AI Similarity Search)** with an `IndexFlatL2` (Euclidean distance) architecture.
* **Weights Integration:** The extractor specifically loads the weights from `task1_model.pth` to ensure the vector space is organized according to features learned during the pneumonia detection task.

---

## 2. Visual Evaluation & Result Analysis
Based on the generated retrieval gallery, the system demonstrates high semantic consistency:

| Query Case | Ground Truth | Retrieval Accuracy (Top 5) | Analysis |
| :--- | :--- | :--- | :--- |
| **Index 0** | Pneumonia | **100% (5/5)** | Consistent low-distance matches ($6.26 - 8.42$). The model successfully matched diffuse opacities. |
| **Index 20** | Normal | **100% (5/5)** | Correctly clustered with clear lung fields. Higher distances ($26.17 - 59.74$) suggest more visual variance in "Normal" signatures. |
| **Index 70** | Pneumonia | **100% (5/5)** | Extremely high precision ($5.40 - 7.10$). Matched specific bilateral textural patterns. |

### Semantic Insights
The low distances for Pneumonia matches indicate that the model has learned a very specific "texture profile" for infection. The higher distances for Normal cases suggest that "absence of disease" is a broader visual category in the embedding space than the presence of specific consolidated patterns.

---

## 3. Clinical Utility
This retrieval engine supports **Case-Based Reasoning (CBR)**. Instead of providing a "black-box" score, it allows a radiologist to:
* Retrieve confirmed historical cases with identical radiographic signatures.
* Validate AI predictions by inspecting the "visual evidence" (nearest neighbors) the model is using to justify its classification.




## 🚀 Project Overview

# Medical AI: Pneumonia Detection, Report Generation, and Retrieval

A multimodal  pipeline for analyzing **PneumoniaMNIST** datasets using CNNs, VLMs (MedGemma), and Vector Databases (FAISS).

## 📁 Project Structure & Deliverables
* **Classification Weights:** `task1_model.pth` (Required for Task 2 & 3)
* **Medical Reports:** `reports/task2_report_generation.md`
* **Retrieval Analysis:** `reports/task3_retrieval.md`
* **Visual Evidence:** `reports/figures/task3_visual_results.png`



This project is divided into three integrated tasks:
1.  **Task 1:** Binary Classification using a custom ResNet-18 CNN.
2.  **Task 2:** Automated Medical Report Generation using MedGemma-4B (VLM) with 4-bit quantization.
3.  **Task 3:** Semantic Image Retrieval using FAISS vector search.

---

## 📦 Requirements & Environment
To replicate this environment, ensure you are using a **T4 GPU** (standard in Google Colab) and install the following:

```text
# General
torch>=2.0.0
torchvision
medmnist
matplotlib
numpy

# VLM & Quantization
transformers>=4.40.0
accelerate>=0.26.0
bitsandbytes>=0.41.1
huggingface_hub
scipy

# Retrieval
faiss-cpu

📂 Model Weights & Integration
The project relies on a modular architecture where Task 2 and Task 3 ingest the weights generated in Task 1.

Requirement: Your trained weights must be saved as task1_model.pth in the root directory.

Usage:

Task 2 loads these weights for side-by-side diagnostic comparison with the VLM.

Task 3 uses the weights to initialize the feature extractor for vector search. 

Device Mapping: The code is optimized with map_location=device to ensure compatibility between local training (Windows/Mac) and cloud inference (Colab).

🔍 Task 3: Semantic Retrieval Summary
The final module implements an image search engine. It enables clinicians to find "visually similar cases" from a historical archive.

Engine: FAISS (Facebook AI Similarity Search).

Latent Space: 512-dimensional embeddings extracted from the ResNet-18 backbone.

Performance: Achieved high precision in retrieving disease-consistent neighbors, as evidenced by the task3_visual_results.png gallery.

🏁 Conclusion
By combining rigorous classification (CNN), human-readable narrative (VLM), and case-based search (FAISS), this pipeline demonstrates a robust, end-to-end clinical decision support system.