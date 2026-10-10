# Sentiment Analysis of Movie Reviews using BERT

**Project 1: NLP Sentiment Classification**
**Author:** [Your name]
**Live demo:** https://huggingface.co/spaces/UnrestrictedTemperament/bert-imdb-sentiment

---

## 1. Introduction

The goal of this project was to build a model that can read a piece of text and decide how the writer feels about it. I worked with movie reviews, and the model's job is to say whether a review is positive or negative. I used BERT (Bidirectional Encoder Representations from Transformers), which is a good fit for this kind of task because it reads a sentence in both directions at once. That helps it pick up on context, such as "not good" meaning the opposite of "good", which older bag-of-words methods often miss.

The end result is a fine-tuned BERT model that reaches about **92% accuracy** on the full IMDB test set, and a web demo that anyone can open in a browser and try without installing anything.

## 2. Dataset

I used the **IMDB Movie Reviews** dataset, loaded straight from the Hugging Face `datasets` library, so no manual download was needed. It contains 25,000 labelled training reviews and 25,000 labelled test reviews, and each review is marked `0` (negative) or `1` (positive). The two classes are balanced.

The project brief mentions positive, negative and neutral categories, and suggests the Twitter Sentiment dataset. I chose IMDB instead because it is a well-known benchmark with clean binary labels, which makes it easy to compare my results against published numbers. The trade-off is that the model has no neutral class, which I come back to in the limitations section.

**How I split the data:**

| Split | Size | Purpose |
|---|---|---|
| Training | 22,500 | Updating the model's weights |
| Validation | 2,500 | Checking progress after each epoch |
| Test | 25,000 | Final, untouched evaluation |

I held out 10% of the original training set as validation (shuffled with a fixed seed of 42) so that the test set stayed completely unseen until the end. At the start I experimented with a small 2,000-review subset to keep runs quick, then switched to the full dataset for the final training.

## 3. Preprocessing

BERT needs its input in a specific format, and the Hugging Face tokenizer handles most of it:

- **Tokenization:** I used the `bert-base-uncased` tokenizer, which lowercases the text and splits it into WordPiece sub-word tokens. This lets BERT handle rare or misspelt words by breaking them into smaller known pieces.
- **Special tokens:** the tokenizer automatically adds `[CLS]` at the start (its final representation is used for classification) and `[SEP]` at the end.
- **Attention masks:** these tell the model which positions are real tokens and which are padding.
- **Truncation:** reviews were cut to a maximum of **256 tokens**. IMDB reviews can be long, and 256 keeps training fast and memory-friendly while still covering most of what a reviewer says. The full BERT limit is 512.
- **Dynamic padding:** instead of padding every review to 256, I used `DataCollatorWithPadding`, which pads each batch only to the length of its longest review. This saves a noticeable amount of computation.

I did not do any manual cleaning such as stripping stop words or punctuation. BERT was pre-trained on natural text, and words like "not" matter a lot for sentiment, so removing them would hurt. The one thing I noticed in the data is that reviews contain HTML line breaks (`<br />`). I left them in, since the tokenizer treats them as a few harmless tokens.

## 4. Model Architecture

I started from the pre-trained **`bert-base-uncased`** checkpoint (12 transformer layers, 768 hidden units, about 110 million parameters) and loaded it with `AutoModelForSequenceClassification`, which puts a classification head on top of BERT.

The head is a single linear layer that takes the final `[CLS]` representation and maps it to two outputs (negative and positive), followed by a softmax to turn them into probabilities. When I loaded the model, the library reported that the classifier weights were newly initialised. That is expected: the BERT body starts with knowledge from pre-training, while the small head starts random, and fine-tuning trains both together on the sentiment task.

## 5. Training Setup

I fine-tuned the model with the Hugging Face `Trainer` on a GPU.

| Setting | Value |
|---|---|
| Base model | bert-base-uncased |
| Learning rate | 2e-5 |
| Batch size | 8 (train and eval) |
| Epochs | 2 |
| Weight decay | 0.01 |
| Max sequence length | 256 |
| Mixed precision (fp16) | On, when a GPU is available |
| Random seed | 42 |
| Evaluation | After every epoch |

I kept the learning rate small (2e-5) because BERT is already well trained, and large updates can erase what it learned in pre-training. I used two epochs since BERT usually converges quickly when fine-tuned, and more epochs on a dataset this size can start to overfit. Weight decay adds mild regularisation, and fp16 speeds up training and reduces memory use.

Training ran for 5,626 steps and took about **16 minutes** on the GPU. The average training loss over the run was about **0.25**.

## 6. Evaluation

After training, I evaluated the model on the full 25,000-review test set, which it had never seen:

| Metric | Result |
|---|---|
| Accuracy | **92.02%** |
| F1-score | **92.05%** |
| Loss | 0.3442 |

An accuracy of about 92% is in line with what is typically reported for BERT-base on IMDB. Accuracy and F1 are almost identical, which tells me the model is not favouring one class over the other.

I also tracked the metrics on the validation set after each epoch:

| Epoch | Training loss | Validation loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.3189 | 0.2977 | 91.04% | 91.22% |
| 2 | 0.0992 | 0.3589 | 91.68% | 91.67% |

Accuracy and F1 improved a little in the second epoch, but the validation loss went up (0.298 to 0.359) while the training loss fell sharply (0.319 to 0.099). That gap is an early sign of **mild overfitting**: the model keeps getting more confident on the training data without getting meaningfully better on new data. It is also why I stopped at two epochs rather than training longer.

The brief asks for precision and recall as well. [FILL IN: precision and recall from the cell in the appendix; paste the values here, plus the confusion matrix if you include it.]

I also tried the model on a few hand-written reviews:

| Review | Prediction | Confidence |
|---|---|---|
| "This movie was absolutely fantastic. Brilliant acting and a gripping story." | Positive | 99.9% |
| "I hated this film. It was boring, too long, and badly written." | Negative | 99.9% |
| "It was okay, some good moments, but overall forgettable." | Negative | 99.4% |

The first two are clear-cut and the model handles them confidently. The third is more interesting and is discussed below.

## 7. Deployment

I wanted people to be able to try the model just by opening a link, so I looked at several ways to host it.

My first attempt was a Gradio interface with a public share link, but Gradio's share-link service failed with an error on my machine. Even when it works, those links expire after a few days and only stay alive while the notebook is running, so they are not a real deployment.

I then moved to **Hugging Face Spaces**. The free tier only supports static sites, so a Python server running the model was not an option. Instead I used **Transformers.js**, which runs the model inside the visitor's own browser:

1. Converted the saved PyTorch model to **ONNX** format with `optimum-cli`.
2. Used the 8-bit quantised version to shrink the download size.
3. Uploaded the ONNX model to a Hugging Face model repository.
4. Built a small web interface (HTML, CSS and JavaScript) as a static Space that loads that model.

The nice side effect is that nothing is sent to a server. The review text stays in the user's browser and the app costs nothing to run. The downside is that the first visit has to download the model, so the page shows a progress bar and a short "please wait" message. After that, the browser caches the model and it loads almost instantly.

The interface shows the verdict, a probability bar for each class, and three clickable example reviews.

## 8. Challenges Faced

- **Gradio share link failing.** Described above. This pushed me towards a proper hosting solution, which turned out to be a better result than I originally planned.
- **Free hosting limits.** Gradio and Docker Spaces need a paid plan, so I had to rethink the deployment around a static site and run the model in the browser.
- **Model conversion errors.** The first ONNX export failed because the tool could not work out the task from a local folder. Passing `--task text-classification` fixed it.
- **Package problems.** A half-updated `huggingface_hub` install broke the upload step, and a forced reinstall plus a kernel restart solved it.
- **Download size.** A full BERT model is large for a web page, so quantisation was necessary to keep the first load reasonable.

## 9. Insights and Limitations

**What I learned**

- Fine-tuning a pre-trained model gets strong results with very little code and about 16 minutes of training. Starting from BERT's existing language knowledge matters much more than building anything from scratch.
- Details that look minor, like dynamic padding, mixed precision and a sensible learning rate, make training much faster and more stable.
- Accuracy alone does not tell the whole story. Watching the validation loss alongside it showed overfitting beginning in the second epoch, even though accuracy was still creeping up.
- Getting a model working in a notebook is only half the job. Packaging it so other people can use it took about as much effort as training it.

**Limitations**

- **Binary output only.** The model has no neutral class. The "okay, some good moments but forgettable" review is genuinely mixed, yet the model called it negative with 99% confidence. This shows that high confidence does not mean the model is right, because it was only ever trained to choose between two options.
- **Domain bias.** It learned from movie reviews, so it may do worse on product reviews, tweets or other kinds of text, where the vocabulary and style are different.
- **Length limit.** Reviews longer than 256 tokens are truncated, so anything said at the end of a long review is ignored.
- **Sarcasm and subtle language.** Like most sentiment models, it can be fooled by sarcasm or by reviews that praise and criticise at the same time.

**Possible improvements**

- Train on a three-class dataset (positive, negative, neutral), or use a star-rating dataset, so mixed reviews have somewhere to go.
- Add early stopping, or a lower learning rate with a learning-rate schedule, since the validation loss started rising in epoch 2. Training longer without that would likely just overfit more.
- Try DistilBERT, which is much smaller and faster and would make the browser demo load quicker.
- Increase the maximum length to 512 tokens for long reviews.
- Fine-tune on a more varied dataset to improve performance outside of movie reviews.

## 10. Conclusion

I fine-tuned BERT on the full IMDB dataset and reached 92.02% accuracy and a 92.05% F1-score on 25,000 unseen reviews. I then deployed it as a free, publicly accessible web app that runs entirely in the browser. The main weakness is that it can only say "positive" or "negative", so extending it to handle neutral sentiment is the most useful next step.

---

## Appendix: Code to compute precision, recall and the confusion matrix

Run this in the notebook after training (it uses the objects that already exist there) and paste the results into Section 6:

```python
from sklearn.metrics import classification_report, confusion_matrix

pred_output = trainer.predict(test_tokenized)
y_pred = np.argmax(pred_output.predictions, axis=1)
y_true = pred_output.label_ids

print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4))
print(confusion_matrix(y_true, y_pred))
```
