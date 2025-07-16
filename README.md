# Hybrid Sentiment Analysis Engine for UGC 
Built to help researchers and marketers extract actionable sentiment insights from noisy, context-heavy online content. A hybrid sentiment analysis engine for UGC, combining LDA topic modeling with Neuro-Symbolic Transformers. Achieves 81% accuracy and outperforms baselines by 14%.

## Purpose
> [!NOTE]
> Sentiment analysis is the task of determining whether a piece of text expresses a positive, neutral, or negative opinion. It’s a key tool for understanding public attitudes and user reactions at scale, especially in fields like marketing, customer service, and content moderation.
> 
> But standard sentiment models struggle when applied to user-generated content (UGC), text written by everyday users on platforms like Twitter, YouTube, or forums. UGC tends to be informal, fragmented, full of slang or emojis, and highly dependent on context. As a result, traditional models often miss subtle tone shifts or misclassify sarcasm and ambiguity.
> 
> This project tackles that problem with a hybrid neuro-symbolic approach. It combines neural transformers (which understand context and language structure) with symbolic topic modeling via LDA (which captures latent themes in the text). By fusing these two types of features, the model can reason about both the meaning and the thematic focus of a sentence, yielding more robust and generalizable sentiment predictions.
>
> The result is a sentiment engine that’s not just accurate, achieving 81% accuracy and outperforming baseline transformers by 14%, but also interpretable and modular, with clear separation between training, evaluation, and deployment logic. It’s designed to help researchers and marketers extract sentiment insights from complex, domain-specific UGC without needing to fine-tune giant models from scratch.

## Overview
The UGC Sentiment Model is a hybrid sentiment classification system that combines deep neural representations with symbolic topic features for improved performance on informal, user-generated content. It integrates ALBERT (a lightweight transformer model) with topic distributions derived from Latent Dirichlet Allocation (LDA), capturing both the contextual meaning and underlying themes of a sentence.

The architecture includes specialized modules for multi-granularity topic extraction (15 and 25 topics), neural feature encoding, and feature fusion. Together, these components enable the model to classify text into positive, neutral, or negative sentiment with greater accuracy and interpretability than standard transformer-based baselines.

This repository includes modular code for data preprocessing, training, evaluation, and inference—designed to support experimentation and easy integration into research or production workflows.

## Results
- **Accuracy**: Achieved 81% classification accuracy on held-out test data, exceeding baseline transformer performance by 14%.
- **Generalization**: Demonstrated improved robustness on user-generated content (UGC), with consistent performance across datasets containing slang, informal grammar, and shifting context.
- **Efficiency**: Supported batch inference in under 300ms per request, with modularized components for training, evaluation, and reuse.
- **Usability**: Included a lightweight UI for running and visualizing predictions, enabling non-technical users to interact with the model and interpret results with ease.
- **Scalability**: Architecture is designed for extensibility—future updates can incorporate new transformer backbones or symbolic inputs without rewriting core logic.

## Prerequisites
> [!IMPORTANT]
> Before installing and running, you'll need: 
> * Python 3.9
> * A CUDA-compatible GPU
> * Hugging Face Account (optional, for hosting/downloading pre-trained models)

## Installation
Follow these steps to set up the environment and install the necessary dependencies.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/KatavinaNguyen/hybrid_sentiment_analysis_engine_for_ugc.git
   cd hybrid_sentiment_analysis_engine_for_ugc
   ```

2. **Create a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers pandas numpy scikit-learn nltk nlpaug imbalanced-learn matplotlib seaborn tqdm huggingface_hub python-dotenv
   ```

4. **Download NLTK Resources**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('opinion_lexicon'); nltk.download('averaged_perceptron_tagger')"
   ```

5. **Set Up Hugging Face Token** (optional)
   ```bash
   echo "HF_READ=" > .env
   ```

## Running the Software

### 1. Data Preprocessing
The preprocessing script (`preprocessing.ipynb`) prepares raw text data for training by cleaning text, extracting LDA features, balancing classes, and tokenizing inputs.

- **Prepare Raw Data**: Place your raw datasets in a directory named `raw_data`. The script expects datasets like Twitter sentiment data, YouTube comments, and climate text data with specific column names. Adjust file paths and column mappings in the script if your data differs.
- **Run Preprocessing**:
   ```bash
   python preprocessing.py
   ```
- **Output**: This generates CSV files (`train_data_balanced.csv`, `val_data.csv`, `test_data.csv`), tokenized NumPy arrays (e.g., `train_input_ids.npy`), LDA features, and pickled objects like `lda_model.pkl` and `sentiment_encoder.pkl`.

### 2. Model Training
The training script (`model train.ipynb`) implements the UGC Sentiment Model, loads preprocessed data, trains the model with mixed precision and early stopping, and evaluates performance on test data.

- **Ensure Preprocessed Data**: Verify that the preprocessing outputs are available in the specified `BASE_PATH` (default: `/kaggle/input/optimized-set`). Update the path in the script if necessary.
- **Run Training**:
   ```bash
   python train.py
   ```
- **Output**: The script saves the best model checkpoint as `hybrid_sentiment_model.pt`, plots training metrics (`training_metrics.png`), and confusion matrices (`confusion_matrix.png`). It also uploads the model to the Hugging Face Hub under `aiguy68/ugc-sentiment-model`.

### 3. Inference (Demo)
The demo script (`demo.py`) demonstrates how to use the pre-trained UGC Sentiment Model for sentiment prediction on new text.

- **Set Up Environment**: Ensure the `.env` file with your Hugging Face token is configured if downloading the model from the Hub.
- **Run Demo**:
   ```bash
   python demo.py
   ```
- **Output**: The script outputs the predicted sentiment (negative, neutral, or positive) and confidence score for the input text (e.g., "I absolutely loved this movie! The acting was superb.").

## Replicating Experiments

### Experiment 1: Data Preprocessing and Class Balancing
- **Objective**: Prepare a balanced dataset for sentiment analysis by cleaning text, augmenting minority classes, and applying SMOTE.
- **Steps**:
  1. Run the preprocessing script as described above.
  2. Verify class distribution in the output `train_data_balanced.csv` to ensure balance across sentiment classes.
- **Expected Results**: The script outputs class distribution statistics before and after balancing, showing improved ratios for minority classes.

### Experiment 2: Multi-Granularity LDA Feature Extraction
- **Objective**: Extract topic distributions at different granularities (15 and 25 topics) to enhance symbolic feature representation.
- **Steps**:
  1. During preprocessing, ensure LDA features are extracted for both 15 and 25 topics (handled automatically in the script).
  2. Check saved NumPy arrays (`train_lda_topics_25.npy`, etc.) for topic distributions.
- **Expected Results**: The preprocessing script logs top words for each topic, indicating thematic coherence in LDA features.

### Experiment 3: Model Training and Evaluation
- **Objective**: Train the UGC Sentiment Model with pre-determined hyperparameters and evaluate performance on validation and test sets.
- **Steps**:
  1. Run the training script as described above.
  2. Monitor training progress via logged metrics (accuracy, F1 score, etc.) and saved plots.
  3. Review test set evaluation metrics printed at the end of training.
- **Expected Results**: The model achieves competitive performance metrics (accuracy, precision, recall, F1) on the test set, with confusion matrices visualizing per-class performance.

### Experiment 4: Inference on New Data
- **Objective**: Test the pre-trained model on unseen text inputs to predict sentiment.
- **Steps**:
  1. Run the demo script with custom text inputs.
  2. Note the predicted sentiment and confidence scores.
- **Expected Results**: The model correctly predicts sentiment for sample inputs, aligning with intuitive sentiment interpretation.

## Configuration and Customization
- **Hyperparameters**: Adjust training parameters like `NUM_EPOCHS`, `LEARNING_RATE`, and `BATCH_SIZE` in the training script (`model train.ipynb`) to experiment with different settings.
- **Model Architecture**: Modify the `HybridSentimentModel` class to experiment with different fusion strategies or transformer backbones (default: `albert-base-v2`).
- **Data Paths**: Update `BASE_PATH` and file paths in scripts to match your local directory structure.

## Troubleshooting
- **GPU Memory Issues**: If you encounter out-of-memory errors, reduce `BATCH_SIZE` or disable mixed precision training by setting `USE_MIXED_PRECISION = False` in the training script.
- **Dependency Conflicts**: Ensure all libraries are installed with compatible versions. Use a clean virtual environment if issues persist.
- **Hugging Face Authentication**: Verify your token in the `.env` file if you face authentication errors while downloading or uploading models.
