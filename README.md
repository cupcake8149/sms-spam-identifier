# SMS Spam Identifier — Web App

A shareable web app built around the Multinomial Naive Bayes spam-classification project in the original ZIP. It has:
- a message tester,
- a performance dashboard (accuracy, precision, recall, confusion matrix, classification report),
- dataset statistics and a sample preview,
- sample messages for a live demonstration.

## Dataset and attribution

The original project notebook uses the UCI SMS Spam Collection dataset and a `CountVectorizer` + `MultinomialNB` model. This app keeps that core approach, while packaging it as a Streamlit application and calculating metrics from its own reproducible, stratified 80/20 train-test split.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset is downloaded when the app starts, so an internet connection is required. The dataset itself is not bundled in this project.

## Easiest way to publish it online (Streamlit Community Cloud)

1. Download and extract `spam_identifier_webapp.zip`.
2. Create a GitHub account if you don't already have one.
3. On GitHub, create a **public repository** named `sms-spam-identifier`.
4. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root. You can also upload the original `LICENSE` file if included.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Click **Create app** / **Deploy an app**.
7. Select your repository, branch `main`, and main file path `app.py`.
8. Click **Deploy**. Wait for the first install and model training to finish.
9. Streamlit will give you a public URL you can share with your teacher.

Do not upload private information or secrets to the public repository. This project needs no API keys.

## Run it on your own computer (optional)

Requires Python 3.10 or newer.

### Windows
1. Install Python from https://www.python.org/downloads/ and enable **Add Python to PATH** during installation.
2. Extract the ZIP and open the extracted folder.
3. In that folder, click the File Explorer address bar, type `cmd`, and press Enter.
4. Run these commands:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. A browser tab should open automatically. If not, visit `http://localhost:8501`.

## How to present it

1. Open the **Check a message** tab.
2. Choose a normal example and click **Check message**.
3. Choose a spam example and run it too.
4. Open **Model dashboard** and explain accuracy, precision, recall, and the confusion matrix.
5. Open **Dataset** to explain the real labelled data used for training.

## Important notes

- The reported scores are calculated when the app runs; they may differ from the original README because this app uses a stratified split.
- A high test score does not mean every new message will be classified correctly.
- “Estimated spam probability” is the model's probability output, not a guarantee or a fully calibrated real-world risk score.
- This is an educational project, not a security product.
