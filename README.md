# 🎬 IMDB Sentiment Classifier with BERT

A BERT model fine-tuned to tell whether a movie review is **positive** or **negative**, with a live demo that runs entirely in your browser.

**👉 Try it here:** [huggingface.co/spaces/UnrestrictedTemperament/bert-imdb-sentiment](https://huggingface.co/spaces/UnrestrictedTemperament/bert-imdb-sentiment)

The first visit downloads the model to your browser, so give it a minute. After that it loads almost instantly, and your text never leaves your device.

## Results

Evaluated on the full IMDB test set (25,000 reviews the model never saw during training):

| Metric | Score |
|---|---|
| Accuracy | **92.02%** |
| F1-score | **92.05%** |
| Loss | 0.3442 |

Training on the full dataset took about 16 minutes on a GPU. For the complete write-up (preprocessing, architecture, per-epoch results, challenges, and limitations), see [`report.md`](report.md).

## How it works

1. **Data:** the IMDB dataset from Hugging Face `datasets` (25,000 train and 25,000 test reviews). I held out 10% of the training set for validation, giving 22,500 / 2,500 / 25,000 for train / validation / test.
2. **Preprocessing:** `bert-base-uncased` tokenizer, reviews truncated to 256 tokens, dynamic padding per batch.
3. **Model:** pre-trained `bert-base-uncased` with a linear classification head (2 classes).
4. **Training:** Hugging Face `Trainer`, learning rate 2e-5, batch size 8, 2 epochs, weight decay 0.01, fp16 on GPU, seed 42.
5. **Deployment:** the model was converted to ONNX and uploaded to Hugging Face. The demo is a static Space that runs it in the browser with [Transformers.js](https://huggingface.co/docs/transformers.js).

## Project structure

```
.
├── Build_an_IMDB_Sentiment_Classifier_using_BERT_with_Gradio_UI.ipynb   # full walkthrough: train, evaluate, save, local Gradio UI
├── train.py            # training script
├── predict.py          # run predictions with the saved model
├── requirements.txt    # Python dependencies
├── sample.csv          # sample data
├── report.md           # detailed project report
└── web_app/            # the browser demo (deployed on Hugging Face Spaces)
    ├── index.html
    ├── index.js
    └── style.css
```

## Run it yourself

**1. Install the dependencies**

```bash
pip install -r requirements.txt
```

PyTorch should be installed separately if you want GPU support. Pick the right command for your system at [pytorch.org](https://pytorch.org/get-started/locally/).

**2. Train the model**

Open the notebook and run all the cells, or run the training script. This saves the fine-tuned model and tokenizer to `./imdb_bert`. A GPU is strongly recommended. On CPU, training on the full dataset will take hours.

**3. Try it locally**

The last cells of the notebook launch a Gradio interface at `http://127.0.0.1:7860`. Open that link in your browser.

> The model weights are not stored in this repo because they are too large for GitHub. Run the notebook to generate them, or use the ONNX version on the Hugging Face Hub: [UnrestrictedTemperament/bert-imdb-sentiment-onnx](https://huggingface.co/UnrestrictedTemperament/bert-imdb-sentiment-onnx).

## Limitations

- **Two classes only.** The model only knows positive and negative, so a genuinely mixed review still gets pushed into one of them, often with high confidence.
- **Movie-review bias.** It was trained on IMDB, so it may be less accurate on other text such as tweets or product reviews.
- **256-token limit.** Anything past the first 256 tokens of a long review is ignored.
- **Mild overfitting.** Validation loss rose slightly in the second epoch while accuracy still improved a little. Early stopping would be a sensible next step.

## Ideas for improvement

- Train on a three-class dataset (positive, negative, neutral) so mixed reviews have somewhere to go.
- Add early stopping or a learning-rate schedule.
- Try DistilBERT for a smaller, faster model and a quicker first load in the browser.
- Fine-tune on more varied data to improve performance outside movie reviews.

## Tech stack

Python · PyTorch · Hugging Face Transformers and Datasets · scikit-learn · Gradio · ONNX · Transformers.js · Hugging Face Spaces
