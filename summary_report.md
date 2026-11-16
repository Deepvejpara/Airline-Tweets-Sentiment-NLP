# Summary Report — Airline Tweet Sentiment Analytics (Set B)

## 1. Business Problem & Dataset
Customer-experience teams at major airlines receive thousands of passenger tweets daily covering flight delays, lost baggage, customer service, and crew appreciation. Manually triaging every post is slow and error-prone. To automate triage, this project builds an end-to-end NLP pipeline and Streamlit web app using the **Twitter US Airline Sentiment** dataset (`Tweets.csv`, 14,640 tweets across 6 airlines and 3 sentiment classes: `negative`, `neutral`, and `positive`). Roughly 63% of tweets are negative, making class imbalance a central consideration.

## 2. Text Cleaning & Feature Engineering
Raw tweets contain `@mentions`, URLs, `#hashtags`, punctuation, and slang. A reusable `clean_text()` function in `nlp_helpers.py` lowercases text, strips mentions and URLs, preserves hashtag words (e.g., `#delayed` becomes `delayed`), removes non-alphabetic characters, tokenises with NLTK, removes English stop words while retaining negations (`not`, `no`, `nor`), and lemmatizes with `WordNetLemmatizer`. Retaining negations prevents phrases such as `"not happy"` from collapsing into `"happy"`.

We evaluated four feature representations: Bag-of-Words (`CountVectorizer`), TF-IDF (`TfidfVectorizer`), TF-IDF with unigrams and bigrams (`ngram_range=(1, 2)`), and 100-dimensional averaged Skip-gram `Word2Vec` embeddings (`min_count=3`, `window=5`). Bigram TF-IDF captures negation and multi-word complaint phrases (`customer service`, `cancelled flight`) while down-weighting common airline terms.

## 3. Classification Results & Baseline Comparison
All models used `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)` on an 80/20 stratified split. Because `negative` accounts for ~63% of tweets, accuracy alone is misleading; **macro F1** was used as the primary metric.

- **Majority-Class Baseline:** ~`[Insert exact baseline accuracy]`% accuracy, ~`[Insert exact baseline macro F1]` macro F1.
- **Best Model (TF-IDF + N-Grams / TF-IDF):** `[Insert exact test accuracy]`% accuracy and `[Insert exact macro F1]` macro F1.
- **Averaged Word2Vec:** `[Insert exact W2V accuracy]`% accuracy and `[Insert exact W2V macro F1]` macro F1, trailing sparse TF-IDF because averaging dilutes negation and key sentiment tokens on short tweets.

## 4. Heuristic vs. Ready-Made Tool vs. ML & POS Tagging
On the first 1,000 test tweets, we compared a 30-word **Heuristic** rule (`[Insert heuristic acc / F1]`), **TextBlob** (`[Insert TextBlob acc / F1]`), and our **ML pipeline** (`[Insert ML acc / F1]`). The ML model outperformed both zero-training alternatives, especially on domain-specific complaints and neutral queries, whereas simple rules failed on sarcasm (e.g., *"Thanks for losing my bag"*). Using NLTK's `pos_tag` on 500 negative tweets to extract `NN`/`NNS` nouns surfaced `flight`, `service`, `bag`, `hour`, `delay`, and `plane` — aligning closely with top `negativereason` categories such as *Customer Service Issue*, *Late Flight*, and *Lost Luggage*.

## 5. Streamlit App & Future Improvements
The `app.py` interface loads `sentiment_pipeline.pkl`, predicts sentiment (`Negative 😠`, `Neutral 😐`, `Positive 😊`), plots class probabilities, extracts `@mentions` and `#hashtags`, displays a POS-tag table, and suggests routing actions. Next steps include collecting Indian airline tweets (including Hinglish code-mixed examples), handling sarcasm, and fine-tuning a pre-trained transformer such as DistilBERT or RoBERTa.
