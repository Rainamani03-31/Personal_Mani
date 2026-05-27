import streamlit as st
import PyPDF2
import nltk
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Page Title
st.set_page_config(page_title="Confidence Threshold Filtering", layout="wide")

st.title("📄 PDF Question Answering System")
st.write("Upload PDF files and ask questions from the documents.")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    documents = []

    # Read PDFs
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text

        documents.append(text)

    st.success(f"Documents Loaded: {len(documents)}")

    # Sentence Tokenization
    sentences = []

    for doc in documents:
        sents = sent_tokenize(doc)
        sentences.extend(sents)

    st.write(f"Total Sentences: {len(sentences)}")

    # Preprocessing
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def preprocess(text):
        tokens = word_tokenize(text.lower())

        clean_tokens = []

        for word in tokens:
            if word.isalpha() and word not in stop_words:
                stem = stemmer.stem(word)
                clean_tokens.append(stem)

        return " ".join(clean_tokens)

    processed_sentences = []

    for s in sentences:
        processed_sentences.append(preprocess(s))

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # User Query
    query = st.text_input("Ask a Question")

    # Threshold & Top K
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

    top_k = st.selectbox("Top K Results", [1, 2, 3, 5], index=2)

    # Search Button
    if st.button("Get Answer"):

        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            processed_query = preprocess(query)

            query_vector = vectorizer.transform([processed_query])

            similarities = cosine_similarity(
                query_vector,
                tfidf_matrix
            )

            scores = similarities[0]

            score_series = pd.Series(scores)

            top_indices = score_series.nlargest(
                top_k
            ).index.tolist()

            # Threshold filtering
            if scores[top_indices[0]] < threshold:
                st.error(
                    "Sorry, I could not find an answer in the documents."
                )
            else:
                st.subheader("Top Matching Answers")

                for idx in top_indices:
                    if scores[idx] >= threshold:
                        st.write("### Answer")
                        st.write(sentences[idx])

                        st.write(
                            f"**Confidence Score:** "
                            f"{scores[idx]:.4f}"
                        )

                        st.divider()

else:
    st.info("Please upload PDF files to continue.")
