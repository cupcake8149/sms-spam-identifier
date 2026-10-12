import io
import zipfile
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="SMS Spam Identifier",
    page_icon="🛡️",
    layout="wide",
)

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


@st.cache_data(show_spinner=False)
def load_dataset():
    """Download the public UCI dataset and read its tab-separated message file."""
    with urlopen(DATASET_URL, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.split("/")[-1] == "SMSSpamCollection"
        ]
        if not candidates:
            raise ValueError("The dataset archive did not contain SMSSpamCollection.")
        with archive.open(candidates[0]) as file:
            df = pd.read_csv(
                file,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
                on_bad_lines="skip",
            )

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df["target"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Training the Naive Bayes model…")
def train_model():
    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.20,
        random_state=1,
        stratify=df["target"],
    )

    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vec, y_train)
    predictions = model.predict(x_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not Spam", "Spam"],
            output_dict=True,
            zero_division=0,
        ),
        "test_count": len(y_test),
        "train_count": len(y_train),
        "predictions": predictions,
        "y_test": y_test,
    }
    return df, vectorizer, model, metrics


st.title("🛡️ SMS Spam Identifier")
st.caption("A machine learning demonstration using Multinomial Naive Bayes and the UCI SMS Spam Collection dataset.")

with st.sidebar:
    st.header("About this project")
    st.write(
        "This app converts message text into word-count features, "
        "then uses a trained Multinomial Naive Bayes classifier to predict "
        "whether a message resembles spam or a normal message."
    )
    st.info(
        "Educational demo only. A prediction is not proof that a message "
        "is safe or fraudulent. Do not click links or share personal data "
        "based only on this result."
    )
    st.markdown("**Dataset:** UCI SMS Spam Collection")
    st.markdown("[View dataset source](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)")

try:
    with st.spinner("Loading the dataset and preparing the model…"):
        df, vectorizer, model, metrics = train_model()
except Exception as exc:
    st.error("The app could not download or prepare the public dataset.")
    st.write("Check your internet connection and try refreshing the app.")
    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
    st.stop()

tab_predict, tab_dashboard, tab_data, tab_about = st.tabs(
    ["🔎 Check a message", "📊 Model dashboard", "🗂️ Dataset", "ℹ️ How it works"]
)

with tab_predict:
    st.subheader("Test a message")
    st.write("Enter a message below, or choose an example to try the model.")

    examples = {
        "Choose an example…": "",
        "Normal: meeting": "Hey, are we still meeting at 5 PM today?",
        "Normal: family": "Mum said dinner is ready. Come home when you can.",
        "Spam: prize": "Congratulations! You have won a free prize. Claim now!",
        "Spam: offer": "URGENT! You have been selected for a cash reward. Call now to claim.",
    }
    selected = st.selectbox("Try a sample message", list(examples.keys()))
    default_message = examples[selected]
    message = st.text_area(
        "Message text",
        value=default_message,
        height=130,
        placeholder="Type or paste an SMS message here…",
        max_chars=5000,
    )

    if st.button("Check message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            vector = vectorizer.transform([message.strip()])
            prediction = int(model.predict(vector)[0])
            probabilities = model.predict_proba(vector)[0]
            spam_index = list(model.classes_).index(1)
            spam_probability = float(probabilities[spam_index])

            if prediction == 1:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: NOT SPAM")

            col1, col2 = st.columns(2)
            col1.metric("Estimated spam probability", f"{spam_probability:.1%}")
            col2.metric("Model vocabulary size", f"{len(vectorizer.vocabulary_):,} words")
            st.progress(spam_probability, text="Model-estimated spam probability")
            st.caption(
                "This probability reflects the model's learned estimates, not a "
                "guarantee of truth or a calibrated real-world risk score."
            )

with tab_dashboard:
    st.subheader("Performance on held-out test data")
    st.write(
        f"The dataset was split into {metrics['train_count']:,} training messages "
        f"and {metrics['test_count']:,} test messages. The test messages were not "
        "used to fit the model."
    )
    a, p, r, n = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    p.metric("Spam precision", f"{metrics['precision']:.2%}")
    r.metric("Spam recall", f"{metrics['recall']:.2%}")
    n.metric("Test messages", f"{metrics['test_count']:,}")

    st.markdown("**Confusion matrix**")
    cm = metrics["matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Not Spam", "Actual: Spam"],
        columns=["Predicted: Not Spam", "Predicted: Spam"],
    )
    st.dataframe(cm_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["Not Spam", "Spam"])
    ax.set_yticks([0, 1], ["Not Spam", "Spam"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Classification report**")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown(
        "- **Accuracy:** share of all test messages classified correctly.\n"
        "- **Precision:** among messages predicted as spam, the share that were actually spam.\n"
        "- **Recall:** among actual spam messages, the share the model detected.\n"
        "- **Confusion matrix:** counts correct and incorrect predictions by category."
    )

with tab_data:
    st.subheader("Dataset overview")
    total = len(df)
    spam_count = int((df["label"] == "spam").sum())
    ham_count = int((df["label"] == "ham").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total messages", f"{total:,}")
    c2.metric("Not spam", f"{ham_count:,}")
    c3.metric("Spam", f"{spam_count:,}")

    counts = df["label"].value_counts().rename(index={"ham": "Not Spam", "spam": "Spam"})
    st.bar_chart(counts)
    st.markdown("**Sample records**")
    st.dataframe(
        df[["label", "message"]].rename(columns={"label": "Original label", "message": "SMS message"}).head(15),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download dataset preview (CSV)",
        data=df[["label", "message"]].head(100).to_csv(index=False).encode("utf-8"),
        file_name="sms_spam_dataset_preview.csv",
        mime="text/csv",
    )

with tab_about:
    st.subheader("How the model works")
    st.markdown(
        """
        1. **Load data:** use SMS messages labelled `ham` (normal) or `spam`.
        2. **Split data:** keep 80% for training and 20% for testing.
        3. **Vectorize text:** `CountVectorizer` converts words into numeric counts.
        4. **Train:** `MultinomialNB` learns patterns associated with each label.
        5. **Evaluate:** calculate accuracy, precision, recall, and the confusion matrix.
        6. **Predict:** transform a new message with the same vectorizer and classify it.

        The split uses a fixed random seed and stratification so the result is repeatable
        and both categories are represented proportionally in train and test sets.
        """
    )
    st.markdown("**Technologies used:** Python, Pandas, Streamlit, Matplotlib, and Scikit-learn.")
    st.markdown(
        "**Important limitation:** this is a traditional text classifier trained on an older "
        "SMS dataset. It can make mistakes on modern scams, new slang, other languages, "
        "or messages unlike its training examples."
    )

st.divider()
st.caption("Student project demonstration • Multinomial Naive Bayes • Results are recalculated from the dataset when the app starts.")
