
# Lecture 1: Introduction to Natural Language Processing
*Instructor: Matei Neagu*

---

## Contents
- Natural language processing: Natural language understanding vs Natural language generation
- Natural language processing applications
- A history of Natural language processing
- Current challenges in Natural language processing
- Conclusions
- Laboratory and marking

---

## Natural Language Processing (NLP)

**NLP** is a field at the intersection of computer science, artificial intelligence, and linguistics, focused on the interactions between computers and human (natural) languages. The need for NLP arises from the vast amount of information recorded in human language. Processing this information with computers can lead to advancements in many fields.

NLP can be broadly divided into two subfields:

- **Natural Language Understanding (NLU):** A subfield of NLP that transforms natural language into formal, structured representations that a computer can process.
- **Natural Language Generation (NLG):** The reverse of NLU, where formal, structured information is transformed into natural language.

---

## NLU Applications

Pure NLU tasks are generally concerned with classifying text into different categories.

### Sentiment Analysis
- **Definition:** Consists of classifying the feeling of a text (e.g., positive, negative, neutral).
- **Domains:**
    - Movie ratings
    - Product review ratings
    - Social media posts
    - Course ratings
    - Medical services

### Spam Detection
- **Definition:** Consists of classifying a text as genuine or spam.
- **Domains:**
    - E-mails
    - Messaging applications
    - Detection of fake reviews

---

## NLG Applications

Most pure NLG tasks take structured data and convert it into text that is easy and pleasant for humans to read.

- **Weather forecasts:** One of the first applications was generating weather reports from weather data.
- **Soccer reports:** Generating text based on the results and statistics of soccer matches.
- **Interactive information about cultural artifacts:** A system proposed for the Buounconsiglo Castle in Trento would automatically generate a post-visit summary for visitors based on their choices.
- **Behavioral change induction:** Generating persuasive letters to encourage people to quit smoking based on their answers to a questionnaire.

---

## General NLP Applications

These applications often involve more complex tasks where both understanding (NLU) and generation (NLG) are needed.

### Automatic Text Summarization
- **Importance:** Crucial due to the massive, exponentially growing amount of textual information online.
- **Objective:** To produce a concise summary of a document (or documents) that is easily consumed by a human reader.
- **Requirements:** Extract all main ideas from the original text while minimizing repetition.

### Question Answering (QA)
- **Definition:** Automatically answering questions asked by a human in natural language, using either a pre-structured database or a collection of documents.
- **Categories:**
    - **Open Domain QA:** Finding an answer from a large corpus like the internet. The goal is to find and synthesize relevant information from multiple sources.
    - **Conversational QA:** Answering questions in a conversational setting, where the context of the prior conversation must be considered.
    - **Machine Reading Comprehension:** The question is provided along with a specific context from which the answer should be extracted.

### Machine Translation
- **Definition:** The task of automatically converting a source text from one language to a target text in another language.

### Prompt-Based Text Generation
- **Definition:** A task where the objective is to produce new text automatically from a given prompt.
- **Types:**
    - Generating natural language text that respects desirable attributes.
    - Generating code based on natural language specifications for software.

### Part-of-Speech (POS) Tagging and Named Entity Recognition (NER)
These are tasks where each word (token) in a text gets a label without needing to interpret the meaning of the whole text. They are outside the core NLU/NLG paradigm but are very important.
- **Part-of-Speech Tagging (POS):** Also called grammatical tagging. It is the automatic assignment of part-of-speech tags (e.g., noun, verb, adverb, pronoun) to words in a sentence.
- **Named Entity Recognition (NER):** The task of identifying mentions of rigid designators from text and classifying them into predefined categories such as person, location, organization, etc.

---

## NLP History

1.  **1930s-1960s:** Early Period - Pure language structure
2.  **1960s-1970s:** AI Introduction
3.  **1970s-1980s:** Language Theory
4.  **1980s-2010s:** Machine Learning
5.  **2010s–2017:** Deep Learning
6.  **2017–Present:** Transformer's Era

### 1. Early Period (1930s-1960s)

#### Summary
- **Focus:** Machine translation, especially between English and Russian (driven by Cold War context).
- **Method:** Characterized by dictionary-to-dictionary translations.
- **Approach:** Focused on morphology, syntax, and semantics, but not the actual *meaning* of words.

#### Highlights
- **1933:** Georges Artsrouni filed a patent for a mechanical translator, the "mechanical brain".
- **WWII:** The Enigma machine can be considered a first practical example of machine translation (decryption).
- **1943-1945:** 'Colossus' was developed to decrypt Enigma.
- **1953:** The Georgetown-IBM experiment automatically translated 60 Russian sentences into English.
- **1957:** Noam Chomsky's *Syntactic Structures* presented grammar in a way that was interpretable by machines.
- **1966:** The **ALPAC report** almost buried the field. Commissioned by US funding agencies, it highlighted the failures of machine translation research, concluding it was easier and cheaper to use human translators.

### 2. AI Introduction (1960s-1970s)

#### Summary
- Built on the development of new Artificial Intelligence techniques.
- For the first time, the **meaning** of words was taken into consideration.
- More emphasis was placed on **world knowledge** stored in knowledge bases.

#### Highlights
- **1961:** **BASEBALL**, a question-answering system that could answer English questions about baseball data stored on a machine.
- **1966:** **ELIZA**, the first chatbot, which simulated a "therapist" using pattern matching and substitution.
- **1968-1970:** **SHRDLU**, a program that allowed a user to instruct a computer to move objects in a virtual "blocks world" using natural language.
- **1969:** Roger Schank introduced the concept of **Tokens** to represent real-world objects, actions, time, and locations.
- **1970:** William Woods introduced **Augmented Transition Networks**, which used finite state automata with recursion to handle complex language structures.
- **1975:** **Minsky's Frame Theory** proposed AI data structures (frames) to represent knowledge using SLOT-VALUE-TYPE pairs (e.g., if ISA is 'person', then NUM_LEGS defaults to 2).

### 3. Language Theory (1970s-1980s)

#### Summary
- This was a grammatical-logical phase.
- Building on augmented transition networks, linguists developed a range of grammar types:
    - **Functional:** (subject, object, etc.)
    - **Categorical:** Combined atomic categories (Sentence, noun) using operators.
    - **Generalised Phrase Structures:** Focused on constituents and subconstituents, oriented towards computability and parsing.
- **1985:** **Generalized Phrase Structured Grammar** became prominent, representing text using tree structures with constituents and subconstituents.

### 4. Machine Learning Models (1980s-2010s)

#### Summary
- Development of ML algorithms and their application to NLP.
- The focus shifted to **linguistic occurrence and co-occurrent patterns** (statistical methods).
- Development of evaluation tools for NLP model performance.

#### Highlights
- **1990:** IBM paper on using **n-grams** for machine translation (English to French).
- **1992:** **Hidden Markov Models (HMMs)** used for speech tagging.
- **1997:** **Long Short-Term Memory (LSTM)** networks developed, with early applications in NLP.
- **1998:** Use of the **Naive Bayes** algorithm for text classification.
- **2001:** **BLEU** (Bilingual Evaluation Understudy) developed as a metric for machine translation.
- **2004:** **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) developed as a metric for summarization and translation.
- **2006:** Google Translate began using a **statistical machine translation** algorithm based on phrases rather than individual words.

### 5. Deep Learning Models (2010s-2017)

#### Summary
- Cheaper GPU power led to an explosion in the use of neural networks for AI problems.
- Large corporations (Google, Facebook, Amazon) began developing production systems using Deep Learning (DL).
- **Recurrent Neural Networks (RNNs)** became increasingly popular for NLP tasks.
- New, useful DL architectures were developed for NLP.

#### Highlights
- **2014:** **Generative Adversarial Networks (GAN)** architecture developed by Ian Goodfellow.
- **2016:** Google Translate moved to **Neural Machine Translation (NMT)**, using an LSTM-based architecture to improve quality.
- **Amazon** released Polly (text-to-speech for Alexa) using LSTMs.
- **2017:** Facebook used LSTM-based networks to perform 4.5 billion automatic translations daily.

### 6. Transformers Era (2017-Present)

#### Summary
- The development of a new deep learning architecture: **Transformers**.
- This architecture took the NLP world by storm, achieving human-level performance on a variety of tasks.

#### Highlights
- **2017:** The pivotal paper **"Attention Is All You Need"** introduced the Transformer architecture and the concept of Large Language Models (LLMs).
- **2019:** The **BERT** model was released.
- **2022:** **ChatGPT** was launched.
- **2024 (Claim):** Goh et al. show above-human-level performance on medical diagnosis by LLMs.
- **2024 (Claim):** ChatGPT passes the Turing test.

---

## Current Challenges

### Costs
- LLMs require massive resources for training and inference, making innovation prohibitive for most researchers.
- Techniques like quantization and flash attention help, but costs remain high.
- Costs also affect users who want to fine-tune LLMs for specific tasks.

### Multilingual LLMs
- Most current efforts and impressive results have focused on English.
- There is a large difference in the volume and quality of training data available for different languages.
- Development of clean, accurate datasets in every language is needed to bring the benefits of LLMs to non-English speakers.

### Language Ambiguity
- Human language ambiguity has not been fully solved by LLMs.
- LLMs struggle with detecting and understanding sarcasm, which requires context beyond the literal text.
- In languages with multiple interpretations for words/phrases (e.g., Chinese, Arabic), models can struggle with nuanced meanings.

---

## Conclusions

- NLP is a fundamental field in computer science with a history as long as that of computers themselves.
- Transformers have revolutionized the field, leading to human-level performance on many tasks.
- There are still significant challenges to tackle, especially related to costs and performance in languages other than English.






# Lecture 2: Text Embeddings
*Instructor: Matei Neagu*

---

## Contents
- Introduction to text representation for NLP
- One hot encoding
- Text cleaning and normalization
- Stemming and lemmatization
- Document level embedding methods: Bag of Words, Bag of N-grams, and TF-IDF
- Practical applications of document level embeddings
- Conclusions

---

## Introduction to Text Representation for NLP

### Challenges in Representing Text for Computers
- **Meaning:** Text does not inherently contain meaning for computers, which operate with numerical representations.
- **Variable Length:** Words and text have variable lengths, while most Machine Learning (ML) methods require fixed-length input.
- **No Natural Order:** There is no intrinsic mathematical order between words (e.g., "bag" > "apples" is meaningless), so simple 1D numerical representations are not suitable.
- **Linguistic "Noise":** Elements like punctuation, conjugation, and articles often add little information for solving NLP tasks and can complicate models.
- **Polysemy:** The same word can have different meanings in various contexts.

### From Text to Vectors: Embeddings
- In language processing, we derive **vectors** from textual data to reflect various linguistic properties. These resulting vectors are called **embeddings**.
- **Tokenization:** This is the act of splitting text into subdivisions called **tokens**. Each token is then transformed into an embedding.
- **Vocabulary:** To represent a text, we need a predefined or learned set of tokens, called a **vocabulary**.
- **What does an embedding vector represent?** The level of tokenization determines this:
    - Word level tokenization
    - Character level tokenization
    - Subword level tokenization
    - Sentence level tokenization
    - Document level embedding

### Tokenization Levels: Pros and Cons
*Example Text: "Ana has apples. She gives one to George."*

#### 1. Word Level Tokenization
- **Example:** `Ana` / `has` / `apples` / `.` / `She` / `gives` / `one` / `to` / `George` / `.`
- **Pros:**
    - Words carry meaning.
    - Produces sequences of a sensible length.
    - Interactions between tokens (words) can be analyzed to discover deeper meaning.
- **Cons:**
    - Very large vocabulary size (Basic English has ~17,000 words).
    - High possibility of **Out-Of-Vocabulary (OOV)** tokens.
    - Misspellings can easily lead to OOV tokens.

#### 2. Character Level Tokenization
- **Example:** `A` / `n` / `a` / ` ` / `h` / `a` / `s` / ` ` / `a` / `p` / `p` / `l` / `e` / `s` / `.` / ...
- **Pros:**
    - Limited vocabulary (e.g., English uses 26 letters plus symbols).
    - Very few OOV tokens.
    - Resistant to misspellings.
- **Cons:**
    - Individual letters do not carry much meaning.
    - Creates very long sequences, as each character becomes a token.

#### 3. Subword Level Tokenization
- **Example:** `A` / `na` / `h` / `as` / `app` / `l` / `es` / `.` / `S` / `he` / `give` / `s` / ...
- **Pros:**
    - Combines the advantages of character and word tokenization.
    - Reduces vocabulary size while keeping meaningful tokens.
    - Reduces the possibility of OOV tokens.
    - Resistant to misspellings.
- **Cons:**
    - Needs a way to handle different possible tokenizations of the same word (e.g., `app/l/es` vs. `apple/s`).

#### 4. Sentence Level Tokenization
- **Example:** `Ana has apples` / `She gives one to George`
- **Pros:**
    - Each token carries significant information.
    - Sequence length is greatly reduced.
- **Cons:**
    - Extremely large, potentially infinite, vocabulary is needed.

#### 5. Document Level Embedding
- **Example:** The entire text "Ana has apples She gives one to George" becomes a single token/vector.
- **Pros:**
    - Sequence length is greatly reduced (to 1).
- **Cons:**
    - Cannot be used to model sequential interactions between words within the document.

---

## One-Hot Encoding
- A classic method to encode categorical data as vectors.
- Given a vocabulary of tokens, each token is encoded as a vector where all values are `0` except for a single `1` at the index corresponding to that token.
- **Example:**
    - Vocabulary: `['ana', 'has', 'apples']`
    - `ana` -> `[1, 0, 0]`
    - `has` -> `[0, 1, 0]`
    - `apples` -> `[0, 0, 1]`
- **Pros:**
    - Simplicity; does not require a complex learning algorithm.
- **Cons:**
    - Memory requirements for each token increase with the size of the vocabulary.
    - Vectors are orthogonal, implying no relationship between words.

---

## Text Cleaning and Normalization
- A key preprocessing step to make raw text data consistent and usable for NLP tasks. It is especially needed for word-level tokenization to reduce the vocabulary size.
- **Techniques:**
    - Hyperlink removal
    - Case normalization
    - Punctuation removal
    - Stop word removal

### Normalization Techniques

#### 1. Hypertext Removal
- **Definition:** Consists of the removal of hypertext (e.g., URLs).
- **Pros:** Removes the need to deal with specific expressions related to URLs (like `https`, `.com`).
- **Cons:** URLs might contain data that is relevant for understanding the text.

#### 2. Case Normalization
- **Definition:** Consists of converting all words to a single case (usually lowercase).
- **Pros:** Reduces the vocabulary size needed to represent words (e.g., "The" and "the" become the same token).
- **Cons:** Case can contain important information. For sentiment analysis, "The accommodation was HORRIBLE" is more intense than "The accommodation was horrible".

#### 3. Punctuation Removal
- **Definition:** Consists of removing punctuation marks from the text.
- **Pros:** Can significantly reduce vocabulary size, as words followed by punctuation (e.g., "word.") would otherwise be treated as different tokens from the word itself.
- **Cons:** May lead to a loss of significant information. Exclamation marks and question marks are important for sentiment analysis, and commas can change the meaning of a sentence.

#### 4. Stop Word Removal
- **Definition:** Consists of removing stop words—common words that have a high frequency but often offer little semantic insight on their own.
- **Examples:** `a`, `the`, `and`, `I`.
- **Pros:**
    - Reduces the length of the text, which helps with processing speed.
    - For RNN-based models, it reduces sequence length, which can help with the vanishing gradient problem.
- **Cons:**
    - Information is lost which might be significant. For example, "I have a bike" and "You have a bike" would both be reduced to "bike", losing the context of ownership.

---

## Stemming and Lemmatization
These are text preprocessing techniques that reduce word variants to a single base form.

- **Stemming:** A cruder, rule-based process that eliminates word affixes (prefixes and suffixes) to reduce words to their common "stem". It's fast but can sometimes produce non-dictionary words.
    - Example: `changing`, `changes`, `changed` -> `chang`
- **Lemmatization:** A more sophisticated process that reduces morphological variants to their dictionary base form (the "lemma"). It considers the word's part of speech and context.
    - Example: `changing`, `changes`, `changed` -> `change`

### Porter Stemmer
- The most popular stemming algorithm, developed by Martin Porter in 1980.
- It removes suffixes based on rules involving the structure of vowels and consonants in a word.
- **Process:**
    1. Words are represented by their vowel (V) and consonant (C) patterns. (e.g., `tree` -> CCVV -> CV).
    2. A word's form is generalized to `[C][VC]{m}[V]`, where `m` is the number of `VC` groups.
    3. It applies a set of rules in 5 sequential steps. The rules are of the form `(condition) S1 -> S2`, where `S1` is a suffix to be replaced by `S2` if the condition on the stem is met. Conditions can be based on `m`, the last letter of the stem, etc.

### Lemmatization Techniques
- **Rule-Based:** Applies predefined rules to find a word's root form.
- **Dictionary-Based:** Uses dictionaries or lookup tables to map words to their lemmas. This often requires a Part-of-Speech (POS) tagger to work correctly.
    - *Example:* **WordNet** (used by NLTK, TextBlob).
- **Machine Learning-Based:** Models learn the rules automatically to reduce a word to its dictionary form.
    - *Example:* **Edit tree lemmatizer** (from SpaCy), which uses Conditional Random Fields.

### Stemming vs. Lemmatization

| Aspect | Stemming | Lemmatization |
| :--- | :--- | :--- |
| **Accuracy** | Less accurate, rule-based, can be imprecise (over/under-stemming). | More accurate, considers context and Part-of-Speech. |
| **Speed** | Faster, uses simple heuristic rules. | Slower, more complex and computationally intensive. |
| **Complexity**| Less complex to implement and understand. | More complex, relies on dictionaries and analyzers. |
| **Flexibility**| Generally language-agnostic. | Depends on language-specific resources (e.g., dictionaries). |

---

## Document Level Embedding Methods

### Bag of Words (BoW)
- A simple method that models a document as an unstructured collection (a "bag") of its words, disregarding grammar and word order but keeping multiplicity.
- **Process:**
    1. Create a vocabulary of all known words.
    2. For each document, create a vector of the same length as the vocabulary.
    3. Each element in the vector corresponds to a word in the vocabulary, and its value is the frequency (count) of that word in the document.
- **Example:**
    - Document: "ana has apples george has mandarins"
    - Vocabulary: `['ana', 'has', 'apples', 'george', 'mandarins', 'strawberries']`
    - BoW Vector: `[1, 2, 1, 1, 1, 0]`
- **Pros:**
    - Simple embedding method.
    - Works well in tasks like text classification.
- **Cons:**
    - **Ignores Context:** Assumes words are independent; loses correlations (`president` is more likely to appear with `election` than `poet`).
    - **Ignores Semantics:** Treats polysemous words (like "bat") as a single token.
    - **Sparsity:** Resulting vectors are very sparse (mostly zeros), which can lead to overfitting.

### TF-IDF (Term Frequency - Inverse Document Frequency)
- An improvement over BoW that weighs words not just by their frequency in one document, but also by how rare they are across all documents in a corpus.
- **Idea:** Words that are frequent in one document but rare in the overall corpus are more important. It mitigates the overrepresentation of common words like "the".
- **Formula:** `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`
    - **Term Frequency (TF):** Measures how frequently a term `t` appears in a document `d`.
        - `TF(t,d) = (count of t in d) / (total tokens in d)`
    - **Inverse Document Frequency (IDF):** Measures how rare a term `t` is across the entire corpus `D`. It penalizes common words.
        - `IDF(t,D) = log( (total documents in D) / (number of documents containing t) )`
- **Pros:**
    - Mitigates the overrepresentation of irrelevant, common words.
- **Cons:**
    - Still suffers from many of the same issues as BoW (ignores word order, context).

### Bag of N-grams (BoN)
- An extension of BoW that captures some local word order and context by using **n-grams** as tokens instead of single words.
- **N-gram:** A continuous sequence of 'N' items (words, characters, etc.) from a text.
    - `1-gram (unigram)`: "ana", "has", "a", ...
    - `2-gram (bigram)`: "ana has", "has a", "a bag", ...
    - `3-gram (trigram)`: "ana has a", "has a bag", ...
- The model builds a "bag" of these n-grams and counts their frequencies.
- **Pros:**
    - Can capture context better than BoW.
    - Alleviates issues of polysemy and improves on capturing word correlations.
    - Versatile: The choice of `N` can be adapted to the task.
- **Cons:**
    - **Curse of Dimensionality:** The vocabulary size explodes as N increases (e.g., 10,000 words -> 100 million possible bigrams).
    - **Sparsity:** The problem of sparse vectors becomes much worse than with BoW.

### Practical Applications of Document-Level Embeddings
- Sentiment analysis
- Spam detection
- Text clustering
- Information retrieval

---

## Conclusions
- A vector representation of text is essential for NLP.
- **Tokenization** breaks text into chunks (words, subwords, etc.) that can be vectorized.
- Text **preprocessing** (cleaning, normalization, stemming, lemmatization) is needed to standardize text, though this might lead to some information loss.
- **Bag of Words**, **TF-IDF**, and **Bag of N-grams** are simple but effective methods for representing entire documents as vectors for various applications.






# Lecture 3: Word Embeddings
*Instructor: Matei Neagu*

---

## Contents
- Limitations of one-hot encoding
- Dense word vectors
- Word2vec
- Co-occurrence of words
- GloVe
- Conclusions

---

## Limitations of One-Hot Encoding
One-hot encoding, while simple, has significant drawbacks for representing words:
- **Scalability:** The embedding vector size increases directly with the vocabulary size, leading to a massive memory footprint for large vocabularies.
- **Sparsity:** Each word is represented by a sparse vector (mostly zeros). This can lead to overfitting when training ML models and is computationally inefficient.
- **Lack of Semantic Meaning:** Words are represented the same regardless of their meaning in different contexts. All word vectors are orthogonal, implying no relationship between any two words.

---

## Dense Word Vectors

Instead of sparse vectors, we can use **dense word vectors**, also known as **word embeddings**.
- Each word is represented as a short, dense vector of floating-point numbers.
- These vectors have a **fixed size** (e.g., 50, 100, 300 dimensions), regardless of the total vocabulary size.
- The goal is for these vectors to capture semantic relationships: words with similar meanings should have similar vectors (i.e., be close to each other in the vector space).

---

## Word2vec

Word2vec is a popular framework for learning these dense word vectors.

### Main Idea
The core principle is that a word's meaning is defined by the company it keeps (**Distributional Hypothesis**).
1.  Start with a large corpus (body) of text.
2.  Assign a random vector to every word in a fixed vocabulary.
3.  Iterate through the text, position by position, using a sliding **window**.
4.  At each position `t`, consider the **center word** `w_t` and the surrounding **context words** `o`.
5.  Use a model to predict the context words from the center word (or vice-versa), calculating a probability.
6.  Continuously adjust the word vectors to maximize this probability, effectively "learning" embeddings that capture context.

### Word2vec Models
There are two main architectures within the Word2vec framework:

#### 1. Continuous Bag-of-Words (CBOW)
- **Task:** Predicts the **center word** from its surrounding context words.
- It essentially takes the "bag of words" in the context window and tries to guess the word in the middle.

#### 2. Skip-gram
- **Task:** Predicts the **context words** from the **center word**.
- Given a center word, it tries to guess the words that are likely to appear nearby.

#### Skip-gram vs. CBOW
- **Training Speed:** CBOW is generally faster to train.
- **Performance on Rare Words:** Skip-gram can generate better embeddings for rare words. This is because CBOW averages the context and is biased towards predicting the most probable word, which is often a common word. Skip-gram, by making multiple predictions for each center word, learns more about specific contexts, including those involving rare words.

### Word2vec Skip-gram: In-depth

#### Loss Function
- For each position `t` in the text, the goal is to predict the context words within a window of size `m`, given the center word `w_t`.
- We want to maximize the likelihood (probability) of observing the true context words around each center word.
- The objective is to minimize the **cost function** `J(θ)`, which is the negative log-likelihood of the predictions.
- **Cost Function:** `J(θ) = - (1/T) * Σ_{t=1 to T} Σ_{-m <= j <= m, j!=0} log(P(w_{t+j} | w_t; θ))`

#### Calculating Probabilities with Softmax
- How do we compute the probability `P(context_word | center_word)`?
- We use **two vectors** for each word `w`:
    - `v_w` when `w` is the center word.
    - `u_w` when `w` is a context word.
- For a center word `c` and a context word `o`, the probability is calculated using the **softmax function**:
    - `P(o|c) = exp(u_o^T * v_c) / Σ_{w∈V} exp(u_w^T * v_c)`
    - The softmax function turns a vector of scores into a probability distribution, amplifying the probability of the highest score.

### Training the Model: Gradient Descent

- The model parameters `θ` (which are all the `u` and `v` vectors) are trained to minimize the cost function `J(θ)`.
- **Gradient Descent** is an optimization algorithm used for this.
    - **Idea:** Start with random parameter values. Repeatedly calculate the gradient (slope) of the cost function and take a small step in the direction of the negative gradient to find the minimum.
    - **Update Equation:** `θ_new = θ_old - α * ∇J(θ)` where `α` is the learning rate.

#### Stochastic Gradient Descent (SGD)
- Calculating the gradient over the entire corpus (billions of windows) for a single update is extremely expensive.
- **Solution:** Use **Stochastic Gradient Descent (SGD)**.
- Instead of the whole corpus, we take a single window (or a small batch of windows) at a time, calculate the gradient for just that sample, and update the weights. This is much faster and more efficient.

### Optimization: Negative Sampling
- A major computational bottleneck in the softmax function is the denominator, which requires summing over the entire vocabulary (often tens of thousands of words) for every prediction.
- **Solution:** **Negative Sampling**.
- **Main Idea:** Instead of a complex multi-class prediction problem, reframe it as a series of simple binary classification problems.
- For a given `(center_word, context_word)` pair:
    1.  Treat the true context word as a **positive sample**.
    2.  Randomly sample `k` other words from the vocabulary that are *not* in the context window. These are the **negative samples** ("noise").
    3.  Train a binary logistic regression model to distinguish the positive sample from the negative samples.
- This means we only need to update the vectors for the words that actually appear in our sample (one positive, `k` negative), which is a huge computational saving.
- **Sampling Probability:** To avoid oversampling frequent words like "the", the probability of picking a word as a negative sample is often adjusted. A common formula is `P(w) = U(w)^(3/4) / Z`, where `U(w)` is the unigram frequency of the word and `Z` is a normalization constant. This helps less frequent words get sampled more often.

### Word2vec: Conclusion
- **Pros:**
    - Creates fixed-size, dense vector representations.
    - The resulting vectors can be used effectively by downstream ML models.
- **Cons:**
    - Assigns only one representation for a word, regardless of its meaning in context (e.g., "bank" of a river vs. a financial "bank").
    - Uses only local information (the context window) to learn representations.
    - Has no built-in way to handle Out-Of-Vocabulary (OOV) words.

---

## Co-occurrence of Words

Word2vec only uses local context. Can we use global context? Yes, by building a **co-occurrence matrix**.

- **Co-occurrence Matrix (X):** A matrix where rows and columns are words from the vocabulary. The cell `X_ij` contains the number of times word `j` appears in the context of word `i`.
- **Context can be defined as:**
    - **Window-based:** Similar to Word2vec, captures syntactic and some semantic information.
    - **Full-document:** Two words co-occur if they appear in the same document. This results in similar representations for words related to the same general topic (e.g., all sports terms will have similar entries).

### Problems with Simple Count Vectors
- **High Dimensionality:** The vectors (rows of the matrix) are the size of the vocabulary, making them very large.
- **Sparsity:** Most entries will be zero, leading to less robust models.

### Dimensionality Reduction with SVD
- **Idea:** Store the most important information in a fixed, small number of dimensions (a dense vector).
- **Method:** Use **Singular Value Decomposition (SVD)** on the co-occurrence matrix `X`.
    - `X = UΣV^T`
    - The rows of the resulting matrix `U` (specifically, the first `k` columns of `U`) can be used as the new, dense, k-dimensional word embeddings.
- **Problem:** Running SVD on raw counts doesn't work well because very frequent function words (the, a, is) dominate the counts.
- **Improvements:**
    - Scale the counts (e.g., take the log, cap the max value).
    - Ignore function words.
    - Use ramped windows where closer words get higher counts.

---

## GloVe: Global Vectors for Word Representation

GloVe combines the strengths of both Word2vec (local context prediction) and matrix factorization (global statistics).

- **Core Idea:** GloVe learns word embeddings from their **global co-occurrence information**, specifically from the *ratio* of co-occurrence probabilities.
- **Insight:** Ratios of co-occurrence probabilities can carry meaning.
    - Let's compare `ice` and `steam`.
    - The ratio `P("solid" | "ice") / P("solid" | "steam")` will be very large.
    - The ratio `P("gas" | "ice") / P("gas" | "steam")` will be very small.
    - The ratio `P("water" | "ice") / P("water" | "steam")` will be close to 1.
    - By training a model to reconstruct these ratios, we can learn meaningful vector embeddings.

### GloVe Loss Function
- The model aims to learn vectors `w_i`, `w_j` and biases `b_i`, `b_j` such that:
    - `w_i^T * w_j + b_i + b_j ≈ log(X_ij)`
    - where `X_ij` is the co-occurrence count of words `i` and `j`.
- The final loss function is a weighted least-squares objective:
    - `J = Σ_{i,j=1 to V} f(X_ij) * (w_i^T * w_j + b_i + b_j - log(X_ij))^2`
- The weighting function `f(X_ij)` serves two purposes:
    1. It is `0` if `X_ij = 0`.
    2. It down-weights very frequent co-occurrences (like stop words) to prevent them from dominating the training.

### GloVe: Conclusion
- **Final Embeddings:** The sum of the two learned word vectors (`U+W`) is often used as the final embedding.
- **Pros:**
    - Creates fixed-size, dense vector representations.
    - Incorporates global statistical information from the entire corpus.
- **Cons:**
    - Like Word2vec, assigns one representation per word, regardless of context.
    - No way to deal with OOV words.

---

## Overall Conclusions

- One-hot encodings are inefficient due to their large size and sparsity.
- **Word embeddings** solve this by creating dense, fixed-size representations of words.
- **Word2vec** uses neural networks to learn embeddings by predicting local word context.
- **Co-occurrence matrices + SVD** can also produce dense embeddings by leveraging global word statistics.
- **GloVe** combines the advantages of both approaches, learning vectors from global co-occurrence statistics in a structured way.







# Lecture 4: ML Methods for NLP Classification
*Instructor: Matei Neagu*

---

## Contents
- NLP classification tasks
- K-NN sentiment analysis
- Naïve Bayes-based sentiment analysis
- SVM-based sentiment analysis
- MLP-based sentiment analysis
- CNN-based sentiment analysis
- RNN-based sentiment analysis
- Measures for classification performance
- Conclusions

---

## NLP Classification Tasks
Classification tasks in NLP involve assigning a predefined category to a piece of text.

### Common Classification Tasks

- **Sentiment Analysis:** Classifying text based on categories related to sentiment.
    - **Binary:** positive/negative
    - **Multiclass:** positive/negative/neutral
    - **Fine-grained/Emotion Detection:** frustration/indifference/shock/disgust/enthusiasm, etc.
- **Spam Detection:** Classifying text (e.g., an email) as spam or not spam based on its content.
- **Topic Labeling:** Categorizing a text based on its topic.
    - **Medical Articles:** cardiology/nephrology/dermatology
    - **News Articles:** sport/science/politics
    - **Books:** bellettristic/technical/self-help
- **Language Identification:** Detecting the language of a given text.
- **Resume Screening:** A binary classification (accepted/rejected) to decide if a candidate's resume is a good fit for a job.
- **Grammar Correctness:** Assessing text as acceptable or not acceptable based on its level of grammatical correctness.

---

## K-NN Based Sentiment Analysis

### K-NN Review
- **K-Nearest Neighbors (K-NN)** is an algorithm that uses proximity to make classifications.
- It is primarily used for classification but can also be used for regression.
- **How it works:**
    1.  Start with a dataset of points, where each point is a vector in a feature space and has an assigned category label.
    2.  For a new, unclassified data point, detect its `k` nearest neighbors based on a chosen distance metric.
    3.  Assign the new point the majority class of its `k` neighbors.
- **Possible Distance Metrics:**
    - **Euclidean Distance:** `D(xi, xj) = sqrt( Σ(x_ik - x_jk)^2 )`
    - **Manhattan Distance:** `D(xi, xj) = Σ |x_ik - x_jk|`
    - **Pearson Correlation:** `D(xi, xj) = 1 - |r(xi, xj)|`
- **Choosing `k`:** The value of `k` is a critical hyperparameter. Small values can lead to overfitting, while large values may lead to underfitting.

### K-NN for Sentiment Analysis
1.  **Get Text Embeddings:** Represent each piece of text (e.g., movie review) as a text-level vector using methods like TF-IDF.
2.  **Assign Labels:** Label each vector in the training set (e.g., positive=1, negative=2, neutral=3).
3.  **Pick a Distance Metric:** Choose a distance like Euclidean.
4.  **Pick `k`:** Choose the number of neighbors to consider.
5.  **Find Neighbors:** For a new text, find its `k` nearest neighbors in the training set.
6.  **Assign Label:** Assign the new text the majority class of its neighbors.

#### Pros of K-NN
- Easy to implement.
- Adapts easily; fast to add new training data.
- Few hyperparameters (only `k` and a distance metric).
- No need to compute model parameters (it's a non-parametric, "lazy" learner).

#### Cons of K-NN
- Does not scale well; memory and storage needs increase with training data size.
- **Curse of Dimensionality:** Performance degrades with high-dimensional data, which is a problem for representations like TF-IDF.
- Prone to overfitting, especially with a small `k`.

---

## Naïve Bayes-Based Sentiment Analysis

### Naïve Bayes Review
- Based on **Bayes' Theorem**: `P(A|B) = (P(B|A) * P(A)) / P(B)`
- In classification terms, we want to find the probability of a label `y` given a feature vector `X`: `P(y|X)`.
- **Naïve Assumption:** The core assumption is that all features `x1, x2, ..., xn` are **independent** of each other.
- This simplifies the formula to: `P(y|X) ∝ P(y) * Π P(xi|y)`
- The term `P(X)` is dropped because it's constant for all classes. We choose the label `y` that maximizes this product.

### Naïve Bayes for Sentiment Analysis
1.  **Get Text Embeddings:** Represent each text as a feature vector. Two popular ways using Bag-of-Words are:
    - **Bernoulli Naïve Bayes:** Features are binary. The vector has a `1` if a word from the vocabulary appears in the text, and `0` otherwise.
    - **Multinomial Naïve Bayes:** Features are integer counts. The vector contains the frequency of each vocabulary word in the text.
2.  **Compute Probabilities:** On the training set, compute:
    - `P(y)`: The prior probability of each class (e.g., what percentage of reviews are positive).
    - `P(xi|y)`: The conditional probability of a feature (word) `xi` occurring, given a class `y`.
3.  **Classify New Text:** For a new text `X`, compute `P(y) * Π P(xi|y)` for each possible class `y` and assign the class that gives the highest score.

#### Pros of Naïve Bayes
- Less complex; simple to compute its parameters.
- Scales well; low storage needs and gives good results when the independence assumption holds.
- Can handle high-dimensional data well.

#### Cons of Naïve Bayes
- **Zero-Frequency Problem:** If a word in a new text never appeared with a certain class in the training data, its probability `P(word|class)` will be zero, making the entire product zero. This is solved with **Laplace smoothing**.
- **Unrealistic Core Assumption:** The feature independence assumption is rarely true in practice for natural language.

---

## SVM-Based Sentiment Analysis

### SVM Review
- **Support Vector Machine (SVM)** is a binary classification model.
- **Goal:** Find an optimal **hyperplane** that separates data points from two classes with the maximum possible **margin**.
- **Hyperplane:** A decision boundary. In 2D it's a line, in 3D it's a plane.
- **Support Vectors:** The data points closest to the hyperplane that define its position.
- **Optimization Problem:** The task is to `Minimize ||w||` (which maximizes the margin `2/||w||`) subject to all data points being correctly classified.

### SVM for Sentiment Analysis
1.  **Get Text Embeddings:** Represent each text as a vector (e.g., TF-IDF, n-grams with TF-IDF).
2.  **Assign Labels:** Assign binary labels to the training data (e.g., positive=+1, negative=-1).
3.  **Find Hyperplane:** Use an optimization algorithm to find the parameters `w` and `b` of the optimal hyperplane.
4.  **Classify New Text:** For a new text vector `x`, classify it using the sign of `w^T*x + b`:
    - `y_hat = +1` if `w^T*x + b >= 0`
    - `y_hat = -1` if `w^T*x + b < 0`

#### Pros of SVM
- **Performance:** Generally provides improved performance compared to simpler methods like K-NN and Naïve Bayes.
- **High-Dimensional Data:** Robust and effective with high-dimensional data like TF-IDF vectors.

#### Cons of SVM
- **Cost:** The optimization process can be computationally costly.
- **Binary Classification:** Naturally a binary classifier. Extending it to multi-class problems requires strategies like one-vs-rest or one-vs-one.

---

## MLP-Based Sentiment Analysis

### MLP Review
- A **Multi-Layer Perceptron (MLP)** is a type of feedforward neural network.
- **Structure:** Consists of an input layer, an output layer, and one or more hidden layers. It is fully connected, meaning nodes in one layer connect to all nodes in the next.
- **Perceptron:** Each node is a perceptron, which computes a weighted sum of its inputs, adds a bias, and then passes the result through an **activation function**.
    - `y_hat = σ(Σ(wi*xi) + w0)`
- **Activation Functions:** Introduce non-linearity, allowing the network to learn complex patterns. Examples: Sigmoid, Tanh, ReLU. For the final output layer in classification, **Softmax** is typically used to create a probability distribution over the classes.
- **Training:** The model is trained by minimizing a **loss function** (e.g., Cross-Entropy for classification) using an **optimization algorithm** (e.g., Gradient Descent, Adam).

### MLP for Sentiment Analysis
1.  **Get Text Embeddings:** Represent each text as a fixed-size vector (e.g., TF-IDF). This vector becomes the input to the MLP.
2.  **Assign Labels:** The labels are one-hot encoded (e.g., Positive=[1,0,0], Neutral=[0,1,0], Negative=[0,0,1]).
3.  **Define Architecture:** Decide on the number of hidden layers, nodes per layer, and activation functions. The output layer will have a number of nodes equal to the number of classes and use a softmax activation.
4.  **Train the Model:** Find the optimal weights `W` using an optimization algorithm.
5.  **Classify New Text:** Feed the new text's embedding through the trained model to get a class prediction.

#### Pros of MLP
- **Performance:** Can achieve good performance on complex problems.

#### Cons of MLP
- **Data Needs:** Requires large datasets for good performance.
- **Dimensionality:** Can be affected by the curse of dimensionality if input vectors are very high-dimensional.
- **Complexity:** Has many hyperparameters to tune (layers, nodes, learning rate, etc.).
- **Cost:** Optimization is costly.

---

## RNN-Based Sentiment Analysis

### RNN Review
- **Recurrent Neural Networks (RNNs)** are designed to work with sequential data.
- **Structure:** They have loops, allowing information to persist. The output of a step `t` depends on the input at `t` and the **hidden state** from the previous step `t-1`.
    - `h_t = f(h_{t-1}, x_t)`
- This "memory" allows RNNs to understand order and context in sequences like text.

### RNN for Sentiment Analysis
1.  **Represent Text as Sequence:** Represent text as a sequence of token-level embeddings (e.g., a sequence of Word2vec or GloVe vectors).
2.  **Define Architecture:** Choose an RNN architecture. For sentence classification, a **many-to-one** architecture is used, where the sequence of word vectors is processed, and the final hidden state is used to make a single prediction.
3.  **Add Classifier:** Add a final MLP layer with a softmax activation on top of the RNN's output to perform the classification.

#### Pros of RNN
- **Sequential Input:** Word order is taken into consideration, allowing the model to capture more complex interactions between words.
- **Performance:** Can achieve good performance, given enough training data.

#### Cons of RNN
- Same cons as MLP (data needs, complexity, cost).
- **Vanishing Gradient:** For long sequences, gradients can become very small during backpropagation, making it difficult for the network to learn long-range dependencies.
- **Lack of Parallelism:** Computation is inherently sequential and cannot be easily parallelized across the time steps.

---

## CNN-Based Sentiment Analysis

### CNN Review
- **Convolutional Neural Networks (CNNs)** were initially developed for computer vision.
- **Structure:** They use **convolutional layers** and **pooling layers** to sequentially reduce dimensionality while capturing local features.
- **Convolution:** A filter (or kernel) slides over the input data, performing a dot product to create a feature map that highlights specific patterns (like n-grams in text).
- **Pooling:** A down-sampling operation (e.g., Max-Pooling) that reduces the size of the feature map, making the model more robust to the exact position of features.

### CNN for Sentiment Analysis
1.  **Represent Text as Matrix:** Represent the text as a matrix where rows are the tokens (sequence length `S`) and columns are the embedding dimensions (`C`). This forms an input matrix of size `S x C`. All texts must be padded to the same sequence length.
2.  **Apply Convolutions:** Use 1D convolutional filters of different sizes (e.g., sizes 2, 3, 4) to act like n-gram detectors, capturing features over 2-word, 3-word, and 4-word phrases.
3.  **Apply Pooling:** Apply a pooling layer (usually max-over-time pooling) to each feature map to get a single value representing the most important feature detected by that filter.
4.  **Add Classifier:** Concatenate the results from the pooling layers and feed them into a final MLP layer for classification.

#### Pros of CNN
- Good performance, as it takes word sequence into account locally.
- Gradients can be computed in parallel, making it more efficient to train than RNNs.

#### Cons of CNN
- Similar cons to MLP (data needs, complexity, etc.).
- Only local information (the size of the filters) is considered at each layer.

---

## Measures for Classification Performance

### Binary Classification
We use a **confusion matrix** to evaluate performance.

|           | Predicted Positive | Predicted Negative |
| :-------- | :----------------- | :----------------- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **Accuracy:** `(TP + TN) / (All)` - Overall correctness. Can be misleading on imbalanced datasets.
- **Precision:** `TP / (TP + FP)` - Of all positive predictions, how many were correct? (Measures exactness).
- **Recall (Sensitivity):** `TP / (TP + FN)` - Of all actual positives, how many did we find? (Measures completeness).
- **F1 Score:** `2 * (Precision * Recall) / (Precision + Recall)` - The harmonic mean of precision and recall. A good single metric for overall performance.

### Multi-Class Classification
- **Accuracy:** Can still be applied: `(Total Correctly Classified) / (Total Datapoints)`.
- For other metrics, we can adapt binary measures.
- **Macro Average:** Calculate the metric (e.g., F1 score) for each class independently and then take the unweighted average. Treats all classes equally.
- **Weighted Average:** Calculate the metric for each class and take the average, weighted by the number of true instances for each class (support). Accounts for class imbalance.
- **Micro Average:** Aggregate the contributions of all classes to compute the average metric. It calculates total TP, FP, and FN across all classes and then computes the metric once.

---

## Conclusions
- Various ML methods can be used for sentiment analysis and other NLP classification tasks.
- The input data can be represented at the **text-level** (for K-NN, Naïve Bayes, SVM, MLP) or at the **token-level as a sequence** (for RNN, CNN).
- More complex models like neural networks generally require more data to perform well.
- A variety of statistical metrics (Accuracy, Precision, Recall, F1) and averaging methods (Macro, Weighted, Micro) are available to evaluate classifier performance for both binary and multiclass cases.




# Lecture 5: Transformers
*Instructor: Matei Neagu*

---

## Contents
- Recurrent Neural Networks (Review)
- Introduction to Transformers
- Transformers Architecture
- Encoders Architecture
- Decoders Architecture
- Hyperparameters
- GPT Family Models
- BERT
- Conclusions

---

## Recurrent Neural Networks (Review)

- From 2010-2017, Recurrent Neural Networks (RNNs) were the go-to method for NLP.
- They are a highly adaptive architecture that works for both generative and classification tasks.
- RNNs reached near human-level performance on tasks like machine translation and sentiment analysis.
- They work by generating a hidden state which is then used as an input in the next step to generate the next hidden state, capturing sequential information.

### Limitations of RNNs
- **Vanishing Gradients:** In long sequences, gradients can become extremely small during backpropagation, meaning the earliest layers in the network learn very little.
- **Long-Term Memory:** The hidden state primarily carries information from the most recent steps. This makes classic RNNs good at short-term memory but very bad at remembering information from much earlier in the sequence.
- **Sequential Processing:** Processing happens one step at a time. Each word must be passed through the network sequentially, and gradients must also be computed sequentially. This prevents parallelization and makes them slow to train on long sequences.

### Enhanced RNNs: LSTM
- **Long Short-Term Memory (LSTM)** networks were developed to address these issues.
- They have 2 hidden states: one for short-term memory and one for long-term memory (the cell state).
- They use three **gates** to control the flow of information:
    - **Forget Gate:** Controls how much of the previous long-term memory to keep.
    - **Input Gate:** Controls how much of the current input to add to the long-term memory.
    - **Output Gate:** Controls what information from the long-term memory is passed to the output.
- **Results:**
    - LSTMs **ameliorate** (but don't perfectly solve) the vanishing gradient problem.
    - LSTMs **ameliorate** the long-term memory issue.
    - LSTMs **do not solve** the sequential processing issue.

---

## Introduction to Transformers

- Introduced in the 2017 paper **"Attention is all you need"**.
- The fundamental mechanism is **attention**.
- They do **not** need recurrence to process sequential interactions.
- Attention is **fully parallel**, solving the sequential processing bottleneck of RNNs.
- They completely solve the long-term memory issue and significantly ameliorate the vanishing gradient problem.

### The Attention Mechanism: Query, Key, Value
Attention is based on the **query-key-value (QKV)** mechanism from information retrieval systems.
- **Query:** What I am looking for.
- **Key:** A label or description of a piece of information.
- **Value:** The actual information.

The process is to match a **Query** with a set of **Keys** to determine how much attention to pay to their corresponding **Values**.

---

## Transformers Architecture

- The classic Transformer is an **encoder-decoder** model.
    - **Encoder:** Takes the input sequence and transforms it into a rich internal representation.
    - **Decoder:** Takes that internal representation and transforms it into an output sequence.
- Both encoders and decoders are neural network-based models.
- A Transformer stack typically has multiple encoders and decoders.
    - All encoders share the same architecture but have different weights.
    - All decoders share the same architecture but have different weights.
    - The number of encoders equals the number of decoders (e.g., 6 of each in the original paper).
- **Encoder Layers:** Each encoder has two main sub-layers: a **Self-Attention** layer and a **Feed-Forward Neural Network (MLP)**.
- **Decoder Layers:** Decoders are similar but have a third sub-layer, an **Encoder-Decoder Attention** layer, which pays attention to the output of the encoder stack.

---

## Encoders Architecture

### Self-Attention Mechanism
This is the core of the Transformer. For each word in the input sequence, self-attention allows it to look at all other words in the same sequence to better understand its context.

1.  **Create Q, K, V Vectors:** For each input word embedding `x_i`, three vectors are created by multiplying it with learned weight matrices:
    - **Query (`q_i`):** `q_i = x_i * W^Q`
    - **Key (`k_i`):** `k_i = x_i * W^K`
    - **Value (`v_i`):** `v_i = x_i * W^V`
2.  **Calculate Attention Scores:** The attention score between every pair of words is calculated by taking the dot product of their Query and Key vectors. `score(q_i, k_j) = q_i · k_j`.
3.  **Scale and Softmax:** The scores are scaled by the square root of the key-vector dimension (`d_k`) to stabilize gradients. A softmax function is then applied to get attention weights, which are probabilities that sum to 1.
4.  **Compute Output Vector:** The output vector `z_i` for each word is a weighted sum of all the Value vectors in the sequence, where the weights are the attention scores calculated in the previous step.

**Formula:** `Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`

- **Example:** In "The **boy** is in the garden. **He** likes to eat ice cream," the self-attention mechanism can learn that "He" refers to "boy," so the new representation of "He" will be heavily influenced by the *value* of "boy," and vice-versa.

### Multi-Headed Attention
- Instead of performing attention once, the Transformer does it multiple times in parallel with different, learned `W^Q`, `W^K`, `W^V` matrices. Each of these is a "head."
- This allows the model to jointly attend to information from different representation subspaces at different positions.
- The outputs of all attention heads are concatenated and then passed through a final linear layer (`W^O`) to produce the final output of the self-attention sub-layer.

### Add & Normalize
- Each sub-layer (Self-Attention and Feed-Forward) has a **residual connection** around it, followed by **layer normalization**.
- `Output = LayerNorm(x + Sublayer(x))`
- **Residual Connection:** The input `x` is added to the output of the sub-layer. This helps prevent the vanishing gradient problem.
- **Layer Normalization:** Normalizes the outputs of the layer to have a mean of 0 and a variance of 1, which helps stabilize and speed up training.

### Feed-Forward Neural Network
- This is a simple, position-wise MLP with two linear layers and a ReLU activation in between.
- `FFN(x) = max(0, x*W1 + b1) * W2 + b2`
- It is applied to each position's vector independently.

### Positional Encoding
- Since the attention mechanism itself has no notion of word order, we must inject positional information.
- This is done by adding a **positional encoding** vector to each input embedding.
- These vectors are not learned; they are generated using fixed `sine` and `cosine` functions of different frequencies.
- This gives the model information about the absolute and relative positions of words in the sequence.

---

## Decoders Architecture
The decoder's job is to generate the output sequence, one token at a time. It has a similar structure to the encoder but with key differences.

### Masked Self-Attention
- The decoder uses a self-attention mechanism, but it is **masked**.
- When predicting the word at position `i`, the decoder is only allowed to attend to words at positions *before* `i`. It cannot "see" future words in the output sequence it is generating. This is crucial for generation tasks.

### Encoder-Decoder Attention
- This is the second attention layer in the decoder.
- It takes the **Query** vectors from the decoder's masked self-attention layer.
- It takes the **Key** and **Value** vectors from the **final output of the encoder stack**.
- This allows every position in the decoder to attend to all positions in the *input sequence*, connecting the two halves of the model.

### Final Output Layer
- After the final decoder block, there is a linear layer followed by a softmax function.
- This produces a probability distribution over the entire vocabulary, and the token with the highest probability is chosen as the next output word.
- This output is then fed back into the decoder as the input for the next time step, generating the sequence token by token.

---

## Hyperparameters
- **N:** Number of encoder/decoder blocks (Original: 6)
- **m:** Embedding size (model_size) (Original: 512)
- **h:** Number of heads in multi-headed attention (Original: 8)
- **d_k, d_v:** Dimension of key and value vectors (Original: 64, since `m/h = 512/8 = 64`)

---

## GPT Family Models (Generative Pre-trained Transformer)
- **Created by:** OpenAI
- **Architecture:** **Decoder-only**. This became a standard for many state-of-the-art generative models.
- **Key Idea:** A decoder-only model can emulate an encoder-decoder architecture using a specific type of attention mask.
- **Impact:** ChatGPT popularized Transformer models, leading to the current revolution in NLP.

- **GPT-1:**
    - 12 heads, 12 blocks.
    - Introduced the paradigm of unsupervised **pre-training** followed by task-specific **fine-tuning**.
    - Fine-tuning involved adding a task-specific "classification head" to the model's output.
- **GPT-2:**
    - Up to 48 blocks.
    - Fused pre-training and fine-tuning, popularizing the concept of **prompting** and **in-context learning** where the model solves tasks without specific fine-tuning.
    - Architectural tweak: Moved layer normalization to the input of each sub-block.
- **GPT-3:**
    - 96 heads, 96 blocks.
    - Introduced alternating dense and sparse attention patterns to manage computational cost.
- **GPT-4:**
    - Introduced **multimodal** capabilities, accepting both text and images as input.

---

## BERT (Bidirectional Encoder from Transformers)
- **Architecture:** **Encoder-only**.
- **Goal:** To produce powerful, general-purpose word and sentence embeddings that can be used for a wide variety of downstream tasks.
- **Key Difference from GPT:** BERT is **bidirectional**. When processing a word, it considers both the left and the right context simultaneously, thanks to its pre-training tasks.

### Pre-training Tasks
1.  **Masked Language Model (MLM):**
    - 15% of the tokens in the input sequence are replaced with a special `[MASK]` token.
    - The model's objective is to predict the original identity of these masked tokens. This forces it to learn a deep bidirectional understanding of language.
2.  **Next Sentence Prediction (NSP):**
    - The model is given two sentences, A and B, and must predict if B is the actual sentence that follows A in the original text or just a random sentence.
    - Special tokens `[CLS]` (at the start) and `[SEP]` (between sentences) are used. The final hidden state of the `[CLS]` token is used for this classification task.

---

## Conclusions

- Transformers are a novel architecture that solves the key issues of parallelism and vanishing gradients found in RNNs.
- The core mechanism is **attention**, which was inspired by retrieval systems.
- Originally, Transformers were encoder-decoder models, but many state-of-the-art variants (like GPT) are **decoder-only**.
- **BERT** is a powerful **encoder-only** model used for generating deep, bidirectional language representations.
- The **GPT family** of decoder-only models has been pivotal in shaping the recent evolution of NLP research.



# Lecture 6: Tokenizers for Transformers
*Instructor: Matei Neagu*

---

## Contents
- Tokenizers Review
- Tokenizers for Transformers
- BPE (Byte-Pair Encoding)
- WordPiece
- Unigram
- SentencePiece
- Conclusions

---

## Tokenizers Review

**Tokenization** is the act of splitting text into subdivisions called **tokens**, which are then transformed into embeddings. The set of all possible tokens is the **vocabulary**.

### Levels of Tokenization

- **Word Level Tokenization**
    - **Pros:** Tokens carry clear meaning; sequences are a sensible length.
    - **Cons:** Very large vocabularies; high chance of Out-Of-Vocabulary (OOV) tokens from misspellings or rare words; requires extensive preprocessing (cleaning, stemming) which can cause information loss.
- **Character Level Tokenization**
    - **Pros:** Limited vocabulary; very few OOV tokens; resistant to misspellings.
    - **Cons:** Individual letters carry little meaning; sequences become significantly longer.
- **Subword Level Tokenization**
    - **Pros:** Combines advantages of word and character levels; reduces vocabulary size while keeping meaningful tokens; reduces OOV possibility; resistant to misspellings.
    - **Cons:** Can produce longer sequences than word-level; needs a consistent way to handle different possible segmentations of the same word (e.g., `apple/s` vs `app/l/es`).
- **Sentence Level Tokenization**
    - **Pros:** Each token carries significant information; sequence length is greatly reduced.
    - **Cons:** Interactions between words within the sentence are lost; vocabulary becomes effectively infinite.

---

## Tokenizers for Transformers

Transformers can accurately capture interactions between elements in long sequences. This has implications for choosing a tokenizer:
- **Granularity is Key:** It's more important to have granular, meaningful representations than short sequences. This makes **sentence tokenizers a poor fit**.
- **Information Loss is Bad:** Traditional preprocessing like stemming and lemmatization, which is necessary for word-level tokenizers to reduce vocabulary size, causes information loss without providing any benefit to the Transformer's performance. This makes **word tokenizers a poor fit**.
- **The Sweet Spot:** **Subword level tokenization** is the best choice. It provides meaningful tokens (better than character-level) and handles the OOV problem (better than word-level) without destructive preprocessing.

### Subword Tokenizer Operations
A subword tokenizer has three main operations:
1.  **Vocabulary Building / Training:** The process of creating the vocabulary of tokens. This is done on a very large corpus to ensure almost no OOV tokens will be encountered later.
2.  **Encoding / Tokenization:** The algorithm used to break input text into a sequence of tokens from the learned vocabulary. There can be multiple valid ways to tokenize a word (e.g., "redemption" -> `red/em/tion` or `red/em/ti/on`).
3.  **Decoding / Detokenization:** The algorithm used to fuse a sequence of generated tokens back into natural language text. This must be handled carefully to avoid ambiguity (e.g., `[the, student, s, nail, ed, ...]` -> "The students nailed..." or "The student snailed...").

### Popular Tokenizers for Transformers
- **Byte-Pair Encoding (BPE):** Used by GPT, GPT-2, RoBERTa, DeBERTa.
- **WordPiece:** Used by BERT, DistilBERT.
- **Unigram:** Used by ALBERT, T5, Big Bird, XLNet.
- **SentencePiece:** A library that efficiently implements Unigram and BPE.

---

## BPE (Byte-Pair Encoding)

- **Origin:** A data compression algorithm from 1994, later adapted for NLP.
- **Main Idea:** The most common words are represented as single tokens, while rare words are broken down into multiple, more frequent subword tokens.

### BPE Vocabulary Building
The process is iterative and bottom-up.
1.  **Pre-tokenize Corpus:** Split the corpus into words (e.g., by spaces). Add a special character (e.g., `Ġ`) to mark word boundaries. Compute the frequency of each word.
2.  **Initialize Vocabulary:** The initial vocabulary consists of all the individual characters present in the pre-tokenized corpus.
3.  **Identify and Merge Frequent Pairs:**
    - Find the most frequent pair of consecutive tokens in the corpus.
    - Merge this pair into a single new token.
    - Add this new token to the vocabulary.
    - Add the merge operation as a "rule" to the ruleset.
4.  **Repeat:** Repeat step 3 until the vocabulary reaches the desired size.

### BPE Tokenization (Encoding)
1.  Break the new string into words, then into its constituent characters.
2.  Iteratively apply the learned merge rules, starting from the lowest-ranked (earliest learned) rule that can be applied to a pair of tokens.
3.  Repeat on the new sequence of tokens until no more merges can be performed.
4.  The final sequence of tokens is the encoded output.

### BPE Detokenization (Decoding)
- Simply concatenate all tokens. The special character `Ġ` is used to know where to reinsert spaces.
- Example: `[Th, e, Ġt, r, a, c, t, o, r, .]` becomes "The tractor."

### BPE Conclusions
- **Pros:**
    - **Simplicity:** The algorithm is straightforward.
    - **Meaningful Tokens:** Creates representative tokens for frequent words.
    - **Flexibility:** Greatly reduces OOV issues by breaking down unknown words.
    - **Consistent:** A given word is always tokenized the same way.
- **Cons:**
    - **Greedy:** Always chooses the most frequent pair, which might not be optimal for the model.
    - **Lack of Morphological Awareness:** Doesn't explicitly use linguistic structure.
    - **Fixed Vocabulary:** The vocabulary is static after training.

---

## WordPiece

- An improvement over BPE, developed by Google and used for BERT.
- The vocabulary building is similar to BPE, but the merge criterion is different.

### WordPiece Vocabulary Building
1.  **Pre-tokenize Corpus:** Same as BPE.
2.  **Initialize Vocabulary:** Start with individual characters. A special prefix (e.g., `##`) is added to characters that are *inside* a word to distinguish them from characters at the start of a word (e.g., "concept" -> `[c, ##o, ##n, ##c, ##e, ##p, ##t]`).
3.  **Identify and Merge Pairs based on Score:**
    - Instead of merging the most frequent pair, WordPiece merges the pair that maximizes a likelihood score.
    - **Score Formula:** `score = freq_of_pair / (freq_of_first_elem * freq_of_second_elem)`
    - This score prioritizes merging pairs where the individual parts are not very common on their own, increasing the chance that the merged token is a meaningful "word piece."
4.  **Repeat:** Repeat step 3 until the vocabulary reaches the desired size.

### WordPiece Tokenization (Encoding)
- This is a top-down, greedy approach.
1.  For a given word, find the longest subword from the vocabulary that is a prefix of the word.
2.  Add this subword to the output.
3.  Move to the rest of the word and repeat until the entire word is tokenized.
4.  If at any point a character is not found in the vocabulary, the word is tokenized as `[UNK]`.

### WordPiece Detokenization (Decoding)
- Simply combine tokens, removing the `##` prefix when concatenating.
- Example: `[H, ##e, h, ##a, ##s, a, h, ##a, ##t]` becomes "He has a hat".

### WordPiece Conclusions
- **Pros:**
    - **Linguistic Relevance:** The scoring method often results in more linguistically motivated subwords.
    - **Vocabulary Size Control:** Manages vocabulary size well.
    - **Improved Performance:** Often leads to better performance, especially in multilingual settings.
- **Cons:**
    - **OOV Words:** Has a higher chance of producing `[UNK]` tokens compared to BPE, which would break an unknown word down to characters.
    - **Computational Complexity:** Can be more computationally intensive than BPE.

---

## Unigram

- A top-down algorithm that relies on a probabilistic model.
- Unlike BPE and WordPiece which build a vocabulary from scratch, Unigram starts with a very large vocabulary and iteratively reduces it.

### Unigram Vocabulary Building
1.  **Initialize a large vocabulary:** This can be done using BPE, WordPiece, or simply by taking all frequent substrings from the corpus.
2.  **Train the Unigram Model:** For the current vocabulary, calculate the probability of each token based on its frequency. Then, for each word in the corpus, find the optimal tokenization by choosing the segmentation that maximizes the product of the probabilities of its tokens.
3.  **Compute Loss:** A loss is computed over the entire corpus based on the likelihood of these optimal segmentations.
4.  **Prune Vocabulary:** For each token in the vocabulary, calculate how much the overall loss would increase if that token were removed.
5.  Remove a certain percentage (e.g., 10%) of the tokens that cause the *smallest* increase in loss (i.e., the least "useful" tokens).
6.  **Repeat:** Repeat steps 2-5 until the vocabulary reaches the desired size.

### Unigram Tokenization (Encoding)
- Given a trained Unigram model (a vocabulary with probabilities), find all possible segmentations of a word.
- The final tokenization is the one with the highest probability (product of its token probabilities).

### Unigram Conclusions
- **Pros:**
    - **Performance:** Not a greedy approach; it evaluates the best tokenization for each word.
    - **Flexibility:** Allows for multiple possible tokenizations of a word, which can be used for data augmentation (subword regularization).
- **Cons:**
    - **Computational Complexity:** Requires extensive computation for training.
    - **Implementation:** May need sophisticated algorithms (like Viterbi) for efficient tokenization.

---

## SentencePiece

- A library that provides a fast and efficient C++ implementation of **Unigram** and **BPE**.
- **Key Advantages:**
    - **Lossless Tokenization:** It treats text as a raw stream of Unicode characters and handles whitespace as a normal character (often represented by `_`). This means the original text can be perfectly reconstructed from the tokens.
    - **Time Performance:** Extremely fast C++ implementation.
    - **On-the-fly Processing:** Can tokenize and detokenize text dynamically during training or inference without pre-tokenizing by spaces.

---

## Overall Conclusions

- **Subword level tokenization** is the best approach for Transformers, offering flexibility for new words while preserving meaningful information.
- **BPE** is a simple, bottom-up algorithm based on merging the most frequent token pairs.
- **WordPiece** improves on BPE by using a likelihood score to guide merges, resulting in more linguistically relevant subwords.
- **Unigram** is a more complex, top-down algorithm that starts with a large vocabulary and prunes it based on a probabilistic loss function, finding the optimal tokenization for each word.




# Lecture 7: Efficient Attention
*Instructor: Matei Neagu*

---

## Contents
- Problem Description
- Efficient Transformers
- Fixed Patterns
- Learnable Patterns
- Low-Rank Methods
- Recurrence Methods
- FlashAttention
- FlashAttention-2
- Conclusions

---

## Problem Description

While Transformers have revolutionized NLP, their performance and computational cost increase with sequence length. This cost is a major limitation, especially with the rise of models supporting very long context lengths (e.g., 128k for GPT-4-Turbo and GPT-4o).

The pricing reflects this cost:
-   **Prompt Tokens:** ~$10.00 / 1 million tokens
-   **Sampled (Output) Tokens:** ~$30.00 / 1 million tokens

### The Attention Bottleneck
The standard attention mechanism is defined as:
`O = softmax( (Q * K^T) / sqrt(d) ) * V`

For a sequence of length `N`, the `Q * K^T` matrix multiplication creates an `N x N` attention score matrix. This leads to:
-   **Time Complexity:** `O(N^2)`
-   **Memory Complexity:** `O(N^2)`

This quadratic complexity is the core problem. The goal is to create more **efficient transformers** by optimizing this attention mechanism.

---

## A Taxonomy of Efficient Transformers

Many methods have been developed to tackle the `O(N^2)` bottleneck. They can be broadly categorized:

-   **Fixed Patterns:** Sparsify the attention matrix using predefined, static patterns.
-   **Learnable Patterns:** Learn the optimal sparse patterns for attention in a data-driven way.
-   **Low-Rank / Kernels:** Use low-rank approximations or kernels to avoid explicitly computing the `N x N` matrix.
-   **Recurrence:** Combine the benefits of attention with recurrence, processing the sequence in segments.
-   **I/O-Aware:** Efficiently use hardware (GPU memory hierarchy) to speed up the exact attention computation without approximation. (e.g., FlashAttention)
-   **Other methods:** Downsampling, Neural Memory, Sparse Models.

---

## Fixed Patterns

These methods sparsify the attention matrix by forcing it to compute attention only for specific, predefined token pairs.

-   **Blockwise Patterns:** Divides the sequence into blocks and computes full attention only within each block. Reduces complexity from `O(N^2)` to `O(N*B)` where `B` is the block size.
-   **Strided Patterns:** Each token attends to other tokens at fixed intervals (strides).
-   **Random Patterns:** Each token attends to a random set of other tokens.
-   **Combined:** Combines multiple fixed patterns to capture both local and global information. **BigBird** is a well-known example that combines blockwise, strided, and random attention.

---

## Learnable Patterns

This is an extension of fixed patterns where the model learns the attention access pattern itself. The key idea is to first determine token relevance and then group tokens into clusters.

### Clustered Attention
- **Idea:** Instead of computing attention for every query (row of the Q matrix), group the queries into `C` clusters.
- **Process:**
    1.  Group the `N` query vectors into `C` clusters (where `C << N`), for example, using K-means clustering.
    2.  Compute a single centroid query vector for each cluster.
    3.  Compute the attention matrix only for these `C` cluster centroids against all `N` keys. This results in a `C x N` attention matrix.
    4.  All original queries within the same cluster use the same attention weights (those of their cluster's centroid) to aggregate the value vectors.
- **Result:** The complexity is reduced from `O(N^2)` to `O(N*C)`.

---

## Low-Rank Methods

- **Assumption:** The `N x N` attention matrix `Q*K^T` is assumed to be **low-rank**, meaning it can be accurately approximated by the product of two smaller matrices.
- **Method:** Decompose or approximate the `N x N` matrix, reducing the problem to `N x k` dimensions, where `k << N`.
- **Example: Linformer**
    - Introduces two linear projection matrices `E` and `F` (`k x N`) that project the `K` and `V` matrices down to a smaller dimension `k`.
    - The attention is then computed as `softmax( Q * (K^T * E) ) * F * V`. The `(K^T * E)` multiplication results in an `N x k` matrix, breaking the quadratic bottleneck.

---

## Recurrence Methods

These methods combine the globality of attention with the linear scaling of RNNs.

- **Idea:** Instead of one large attention computation over a sequence of length `N`, break the sequence into smaller segments and process them with an attention mechanism connected by recurrence.
- **Transformer-XL:** A prominent example that introduces a **segment-level recurrence** mechanism.
    - It processes the input segment by segment.
    - When processing the current segment, it also attends to the hidden states from the *previous* segment, which are cached in a memory.
    - This allows the model to learn dependencies far beyond the length of a single segment.
    - It uses **relative positional encodings** instead of absolute ones, which is more suitable for this recurrence structure.
- **Advantages:** Combines the perfect token-to-token connection of attention with the linear cost increase of RNNs. It can learn dependencies 80% longer than standard RNNs and 450% longer than vanilla Transformers.

---

## FlashAttention: I/O-Aware Attention

FlashAttention doesn't approximate attention; it computes the **exact** attention mechanism but does so much faster by being "I/O-aware," meaning it intelligently manages the GPU memory hierarchy.

### GPU Memory Hierarchy
- **SRAM:** Very fast on-chip memory (e.g., 19 TB/s), but very small (e.g., 20 MB).
- **HBM (High-Bandwidth Memory):** Slower off-chip GPU memory (e.g., 1.5 TB/s), but much larger (e.g., 40 GB).
- **Bottleneck:** The main bottleneck for standard attention is not computation (FLOPS) but memory bandwidth—the time it takes to read and write the large `N x N` matrices to and from HBM.

### FlashAttention's Approach
FlashAttention restructures the attention computation to minimize reads/writes from the slow HBM by maximizing use of the fast SRAM.

-   **Tiling:** The algorithm breaks the large Q, K, and V matrices into smaller blocks or "tiles." It loads these tiles from HBM into the fast SRAM, performs the attention computation for that tile, and writes the result back. This keeps most of the work on-chip.
-   **Recomputation:** To save memory, FlashAttention does not store the large `N x N` attention matrix for the backward pass (gradient computation). Instead, it stores the output and softmax normalization statistics from the forward pass and uses them to **recompute** the necessary attention values on the fly during the backward pass. This clever trick reduces the memory complexity from `O(N^2)` to `O(N)`.
-   **Online Softmax:** It uses a numerically stable "online softmax" algorithm that can compute the softmax of tiled inputs in a single pass without needing to access the full row of the attention matrix.

### FlashAttention Results
- Extremely fast, especially for long sequences.
- Can handle much longer sequences due to the `O(N)` memory reduction.
- Fast even when computing full (non-sparse) attention.

---

## FlashAttention-2

FlashAttention-2 is an improvement over the original, optimizing performance even further.

-   **Reduces Redundant Operations:** Streamlines the algorithm to eliminate unnecessary memory reads/writes.
-   **Improved Parallelism:** Parallelizes the computation not just over the batch size and number of heads, but also over the **sequence length axis**. This is achieved by changing the loop structure—the outer loop is over blocks of Q, while the inner loop is over blocks of K and V. This keeps thread workloads more balanced and reduces synchronization needs.
-   **Better Hardware Utilization:** The changes lead to even better hardware utilization, especially on newer GPUs like the H100.
-   **Results:**
    -   ~2x faster than FlashAttention.
    -   Up to 10x faster than standard attention implementations.
    -   Scales incredibly well with better hardware.

---

## Conclusions

-   The standard attention mechanism is the bottleneck in Transformers, with `O(N^2)` time and memory complexity, making long sequences very costly.
-   Most methods to solve this issue **approximate** the attention matrix (e.g., sparse patterns, low-rank methods).
-   **FlashAttention** and **FlashAttention-2** provide a different solution: they compute the **exact** attention but restructure the algorithm to be I/O-aware, drastically reducing memory transfers and increasing speed.
-   FlashAttention achieves this through **tiling** and **recomputation**, making memory requirements linear (`O(N)`).
-   FlashAttention-2 further optimizes by improving **parallelism** and reducing redundant operations, achieving significant speedups.



# Lecture 8: Similarity Learning
*Instructor: Matei Neagu*

---

## Contents
- Introduction to Similarity Learning
- Similarity with Predefined Embeddings
- Siamese Networks
- Sentence-BERT
- Conclusions

---

## Introduction to Similarity Learning

**Similarity Learning** is a specialized area of supervised machine learning. Unlike traditional supervised learning which focuses on predicting a label for a single input, similarity learning focuses on measuring the **similarity or dissimilarity** between two or more data points.

-   **Core Concept:** The goal is to learn an embedding space where similar items are mapped to nearby points and dissimilar items are mapped to distant points.
-   **NLP Context:** Two pieces of text might use different words or sentence structures but convey the same meaning. Similarity learning aims to capture this semantic equivalence.
-   **Use Case:** It is particularly useful in scenarios with a high number of classes but very few training examples per class (e.g., face verification), where classic classification methods tend to fail.

### Applications
-   **Recommendation Systems:** Find content (articles, products, music) that is similar to what a user has already liked.
-   **Face Verification:** Compare the facial features in an image against a database of known faces to find a match.
-   **Anomaly Detection:** By learning what "normal" data looks like in the embedding space, any point that maps far away from normal clusters can be flagged as an anomaly.

---

## Similarity with Predefined Embeddings

A straightforward approach to measuring text similarity without training a specialized model.

### Process
1.  Take two pieces of text.
2.  Use a classic text embedding method (like TF-IDF) to represent each text as a vector.
3.  Compute a distance or similarity score between the two vectors.
4.  Use this score to make a decision (e.g., apply a threshold to decide if they are the same) or to rank items (e.g., retrieve the top-k most similar texts for a recommender system).

### Distance Metrics

-   **Euclidean Distance:** The straight-line distance between two vectors in the embedding space.
    -   **Disadvantage:** For word frequency embeddings (like TF-IDF), longer documents will naturally have larger vectors and thus larger Euclidean distances, even if they are semantically similar to shorter documents.
-   **Cosine Similarity/Distance:** Measures the cosine of the angle between two vectors. It is independent of vector magnitude.
    -   `Cosine Distance = 1 - Cosine Similarity`
    -   **Advantage:** This is the most common metric for text similarity as it mitigates the issue of text length. Texts with similar word frequencies will have a small angle (high similarity) regardless of their overall length.
-   **Jaccard Distance:** Measures dissimilarity between sets. It does not require embeddings.
    -   `Jaccard Distance = 1 - ( |Set1 ∩ Set2| / |Set1 ∪ Set2| )`
    -   **Advantage:** It is computationally cheap and is well-suited for applications where the raw count of words is not important, only their presence or absence (e.g., comparing product descriptions).
-   **Word Mover's Distance (WMD):** Builds upon word embeddings (like Word2vec). It measures the minimum "cost" required to "travel" from the words in one document to the words in another. It provides strong performance without needing hyperparameters.

---

## Siamese Networks

While predefined embeddings can work, they are not optimized for a specific similarity task. To improve performance, we can learn embeddings that are specific to the task at hand.

-   **Solution:** **Siamese Networks**.

### Architecture
A Siamese network consists of:
1.  **Two inputs:** These are the two items to be compared (e.g., two pieces of text, two images).
2.  **Two identical networks with shared weights:** Both inputs are passed through two separate network "towers" that have the exact same architecture and parameters. Sharing weights ensures that both inputs are projected into the same embedding space.
3.  **Two embedding outputs:** Each tower outputs an embedding vector for its input.
4.  **A loss function:** This function takes the two embedding outputs and a label (indicating if the original inputs were similar or dissimilar) and computes a loss value. The network is trained to minimize this loss.

-   **Network Components:**
    -   **Embedding Layer:** Converts tokens to word embeddings (Word2vec, GloVe, etc.).
    -   **Encoder Network:** Can be an RNN, CNN, or Transformer-based model.
    -   **Loss Function:** A specialized loss function is needed. While classic distances can be used, **Contrastive Loss** and **Triplet Loss** were developed specifically for this purpose.

### Contrastive Loss

-   **Goal:** To pull embeddings of similar items closer together and push embeddings of dissimilar items apart.
-   **Mechanism:** It defines a **margin `m`**.
    -   For a **similar pair**, the loss is simply their squared Euclidean distance. The goal is to minimize this distance (pull them together).
    -   For a **dissimilar pair**, the loss is zero if their distance is greater than the margin `m`. If their distance is less than `m`, the loss function penalizes the model, pushing them apart until their distance is at least `m`.
-   **Benefit:** The margin `m` prevents the model from pushing dissimilar pairs infinitely far apart, which creates a more stable and well-structured embedding space.

### Triplet Loss

-   **Goal:** To ensure that a "positive" example is closer to an "anchor" example than a "negative" example is, by at least a certain margin.
-   **Mechanism:** The network is trained on **triplets** of data points:
    1.  **Anchor (`A`):** A reference data point.
    2.  **Positive (`P`):** A data point from the same class as the anchor.
    3.  **Negative (`N`):** A data point from a different class than the anchor.
-   **Loss Formula:** `Loss = max( d(A, P)^2 - d(A, N)^2 + α, 0 )`
    -   `d(A, P)` is the distance between the anchor and the positive.
    -   `d(A, N)` is the distance between the anchor and the negative.
    -   `α` is the margin.
-   The loss is zero only if `d(A, N)` is greater than `d(A, P)` by at least the margin `α`.

### Triplet Loss vs. Contrastive Loss
-   Triplet loss directly compares positive and negative pair distances; contrastive loss handles them separately.
-   Contrastive loss tends to force similar points to collapse to a distance of 0, while Triplet loss allows for more variance within a class cluster.
-   Empirically, Triplet Loss can continue to organize the vector space for longer, potentially reaching a better state, while Contrastive Loss might converge to a local minimum earlier.

### Triplet Loss Implementation
-   **Naive "Offline" Strategy:** Form all possible triplets at the start of an epoch and train on them. This is inefficient as many triplets are "easy" (the negative is already very far away) and provide no learning signal.
-   **Efficient "Online" Strategy:**
    1.  Take a batch of `n` samples.
    2.  Compute all `n` embeddings.
    3.  Form valid triplets "on the fly" from within the batch.
    4.  Only use "hard" or "semi-hard" triplets (where the negative is closer to the anchor than the positive is) to compute the loss and update the weights.

---

## Sentence-BERT (SBERT)

The Siamese network concept can be applied to modern Transformer architectures.
-   For this task, an **encoder-only** architecture like BERT is a better fit than a full encoder-decoder model.
-   **Sentence-BERT** combines the **BERT** architecture with the **Siamese** idea to produce semantically meaningful sentence embeddings.

### BERT for Sentence Comparison (The Problem)
-   Using a raw, pre-trained BERT for sentence similarity is computationally very expensive and often yields poor results.
-   Feeding both sentences to BERT to get a classification is extremely slow for finding the most similar pair in a large collection.
-   Computing individual sentence embeddings from BERT (e.g., by averaging word embeddings or using the `[CLS]` token embedding) does not produce good results for similarity tasks out-of-the-box. The `[CLS]` token, in particular, is not optimized for this.

### Sentence-BERT (The Solution)
-   **Architecture:** A Siamese network that uses two identical, weight-sharing BERT models as its towers.
-   **Process:**
    1.  Two sentences are passed through their respective BERT models.
    2.  A **pooling** operation is applied to the output token embeddings of each BERT to produce a single, fixed-size sentence embedding. (Mean-pooling over all tokens often works best).
    3.  These two sentence embeddings are then used with a specific loss function to fine-tune the BERT weights.
-   **Objectives (Training SBERT):**
    1.  **Classification Objective:** The two sentence embeddings `u` and `v` are concatenated (often with their difference `|u-v|`) and passed to a simple MLP with a softmax output. This is trained on datasets like NLI (Natural Language Inference).
    2.  **Regression Objective:** A similarity metric (like cosine similarity) is computed between `u` and `v`. The model is trained using MSE loss to make this score match a human-annotated similarity score (e.g., from the STS-B dataset).
    3.  **Triplet Objective:** Three sentences (anchor, positive, negative) are passed through the network to get three embeddings, and Triplet Loss is used to update the weights.

### SBERT Performance
-   SBERT performs significantly better than non-trainable methods (like averaging GloVe embeddings) and raw BERT embeddings.
-   Average-pooling over the output tokens is a very effective strategy.
-   Explicitly concatenating the difference vector `|u-v|` with the sentence vectors `u` and `v` for the classification objective significantly improves performance.

---

## Conclusions

-   Classic classification methods often fail in scenarios with many classes but little data per class.
-   **Similarity Learning** is a powerful technique to solve this problem by learning a meaningful embedding space.
-   Similarity methods can be **non-trainable** (using predefined embeddings and distance metrics) or **trainable**.
-   The most popular trainable architecture is the **Siamese Network**.
-   Two special loss functions, **Contrastive Loss** and **Triplet Loss**, were invented to effectively train these networks.
-   **Sentence-BERT** successfully combines the Siamese network paradigm with the power of the Transformer architecture to create state-of-the-art sentence embeddings for similarity tasks.





# Lecture 8: Similarity Learning
*Instructor: Matei Neagu*

---

## Contents
- Introduction to Similarity Learning
- Similarity with Predefined Embeddings
- Siamese Networks
- Sentence-BERT
- Conclusions

---

## Introduction to Similarity Learning

**Similarity Learning** is a specialized area of supervised machine learning. Unlike traditional supervised learning which focuses on predicting a label for a single input, similarity learning focuses on measuring the **similarity or dissimilarity** between two or more data points.

-   **Core Concept:** The goal is to learn an embedding space where similar items are mapped to nearby points and dissimilar items are mapped to distant points.
-   **NLP Context:** Two pieces of text might use different words or sentence structures but convey the same meaning. Similarity learning aims to capture this semantic equivalence.
-   **Use Case:** It is particularly useful in scenarios with a high number of classes but very few training examples per class (e.g., face verification), where classic classification methods tend to fail.

### Applications
-   **Recommendation Systems:** Find content (articles, products, music) that is similar to what a user has already liked.
-   **Face Verification:** Compare the facial features in an image against a database of known faces to find a match.
-   **Anomaly Detection:** By learning what "normal" data looks like in the embedding space, any point that maps far away from normal clusters can be flagged as an anomaly.

---

## Similarity with Predefined Embeddings

A straightforward approach to measuring text similarity without training a specialized model.

### Process
1.  Take two pieces of text.
2.  Use a classic text embedding method (like TF-IDF) to represent each text as a vector.
3.  Compute a distance or similarity score between the two vectors.
4.  Use this score to make a decision (e.g., apply a threshold to decide if they are the same) or to rank items (e.g., retrieve the top-k most similar texts for a recommender system).

### Distance Metrics

-   **Euclidean Distance:** The straight-line distance between two vectors in the embedding space.
    -   **Disadvantage:** For word frequency embeddings (like TF-IDF), longer documents will naturally have larger vectors and thus larger Euclidean distances, even if they are semantically similar to shorter documents.
-   **Cosine Similarity/Distance:** Measures the cosine of the angle between two vectors. It is independent of vector magnitude.
    -   `Cosine Distance = 1 - Cosine Similarity`
    -   **Advantage:** This is the most common metric for text similarity as it mitigates the issue of text length. Texts with similar word frequencies will have a small angle (high similarity) regardless of their overall length.
-   **Jaccard Distance:** Measures dissimilarity between sets. It does not require embeddings.
    -   `Jaccard Distance = 1 - ( |Set1 ∩ Set2| / |Set1 ∪ Set2| )`
    -   **Advantage:** It is computationally cheap and is well-suited for applications where the raw count of words is not important, only their presence or absence (e.g., comparing product descriptions).
-   **Word Mover's Distance (WMD):** Builds upon word embeddings (like Word2vec). It measures the minimum "cost" required to "travel" from the words in one document to the words in another. It provides strong performance without needing hyperparameters.

---

## Siamese Networks

While predefined embeddings can work, they are not optimized for a specific similarity task. To improve performance, we can learn embeddings that are specific to the task at hand.

-   **Solution:** **Siamese Networks**.

### Architecture
A Siamese network consists of:
1.  **Two inputs:** These are the two items to be compared (e.g., two pieces of text, two images).
2.  **Two identical networks with shared weights:** Both inputs are passed through two separate network "towers" that have the exact same architecture and parameters. Sharing weights ensures that both inputs are projected into the same embedding space.
3.  **Two embedding outputs:** Each tower outputs an embedding vector for its input.
4.  **A loss function:** This function takes the two embedding outputs and a label (indicating if the original inputs were similar or dissimilar) and computes a loss value. The network is trained to minimize this loss.

-   **Network Components:**
    -   **Embedding Layer:** Converts tokens to word embeddings (Word2vec, GloVe, etc.).
    -   **Encoder Network:** Can be an RNN, CNN, or Transformer-based model.
    -   **Loss Function:** A specialized loss function is needed. While classic distances can be used, **Contrastive Loss** and **Triplet Loss** were developed specifically for this purpose.

### Contrastive Loss

-   **Goal:** To pull embeddings of similar items closer together and push embeddings of dissimilar items apart.
-   **Mechanism:** It defines a **margin `m`**.
    -   For a **similar pair**, the loss is simply their squared Euclidean distance. The goal is to minimize this distance (pull them together).
    -   For a **dissimilar pair**, the loss is zero if their distance is greater than the margin `m`. If their distance is less than `m`, the loss function penalizes the model, pushing them apart until their distance is at least `m`.
-   **Benefit:** The margin `m` prevents the model from pushing dissimilar pairs infinitely far apart, which creates a more stable and well-structured embedding space.

### Triplet Loss

-   **Goal:** To ensure that a "positive" example is closer to an "anchor" example than a "negative" example is, by at least a certain margin.
-   **Mechanism:** The network is trained on **triplets** of data points:
    1.  **Anchor (`A`):** A reference data point.
    2.  **Positive (`P`):** A data point from the same class as the anchor.
    3.  **Negative (`N`):** A data point from a different class than the anchor.
-   **Loss Formula:** `Loss = max( d(A, P)^2 - d(A, N)^2 + α, 0 )`
    -   `d(A, P)` is the distance between the anchor and the positive.
    -   `d(A, N)` is the distance between the anchor and the negative.
    -   `α` is the margin.
-   The loss is zero only if `d(A, N)` is greater than `d(A, P)` by at least the margin `α`.

### Triplet Loss vs. Contrastive Loss
-   Triplet loss directly compares positive and negative pair distances; contrastive loss handles them separately.
-   Contrastive loss tends to force similar points to collapse to a distance of 0, while Triplet loss allows for more variance within a class cluster.
-   Empirically, Triplet Loss can continue to organize the vector space for longer, potentially reaching a better state, while Contrastive Loss might converge to a local minimum earlier.

### Triplet Loss Implementation
-   **Naive "Offline" Strategy:** Form all possible triplets at the start of an epoch and train on them. This is inefficient as many triplets are "easy" (the negative is already very far away) and provide no learning signal.
-   **Efficient "Online" Strategy:**
    1.  Take a batch of `n` samples.
    2.  Compute all `n` embeddings.
    3.  Form valid triplets "on the fly" from within the batch.
    4.  Only use "hard" or "semi-hard" triplets (where the negative is closer to the anchor than the positive is) to compute the loss and update the weights.

---

## Sentence-BERT (SBERT)

The Siamese network concept can be applied to modern Transformer architectures.
-   For this task, an **encoder-only** architecture like BERT is a better fit than a full encoder-decoder model.
-   **Sentence-BERT** combines the **BERT** architecture with the **Siamese** idea to produce semantically meaningful sentence embeddings.

### BERT for Sentence Comparison (The Problem)
-   Using a raw, pre-trained BERT for sentence similarity is computationally very expensive and often yields poor results.
-   Feeding both sentences to BERT to get a classification is extremely slow for finding the most similar pair in a large collection.
-   Computing individual sentence embeddings from BERT (e.g., by averaging word embeddings or using the `[CLS]` token embedding) does not produce good results for similarity tasks out-of-the-box. The `[CLS]` token, in particular, is not optimized for this.

### Sentence-BERT (The Solution)
-   **Architecture:** A Siamese network that uses two identical, weight-sharing BERT models as its towers.
-   **Process:**
    1.  Two sentences are passed through their respective BERT models.
    2.  A **pooling** operation is applied to the output token embeddings of each BERT to produce a single, fixed-size sentence embedding. (Mean-pooling over all tokens often works best).
    3.  These two sentence embeddings are then used with a specific loss function to fine-tune the BERT weights.
-   **Objectives (Training SBERT):**
    1.  **Classification Objective:** The two sentence embeddings `u` and `v` are concatenated (often with their difference `|u-v|`) and passed to a simple MLP with a softmax output. This is trained on datasets like NLI (Natural Language Inference).
    2.  **Regression Objective:** A similarity metric (like cosine similarity) is computed between `u` and `v`. The model is trained using MSE loss to make this score match a human-annotated similarity score (e.g., from the STS-B dataset).
    3.  **Triplet Objective:** Three sentences (anchor, positive, negative) are passed through the network to get three embeddings, and Triplet Loss is used to update the weights.

### SBERT Performance
-   SBERT performs significantly better than non-trainable methods (like averaging GloVe embeddings) and raw BERT embeddings.
-   Average-pooling over the output tokens is a very effective strategy.
-   Explicitly concatenating the difference vector `|u-v|` with the sentence vectors `u` and `v` for the classification objective significantly improves performance.

---

## Conclusions

-   Classic classification methods often fail in scenarios with many classes but little data per class.
-   **Similarity Learning** is a powerful technique to solve this problem by learning a meaningful embedding space.
-   Similarity methods can be **non-trainable** (using predefined embeddings and distance metrics) or **trainable**.
-   The most popular trainable architecture is the **Siamese Network**.
-   Two special loss functions, **Contrastive Loss** and **Triplet Loss**, were invented to effectively train these networks.
-   **Sentence-BERT** successfully combines the Siamese network paradigm with the power of the Transformer architecture to create state-of-the-art sentence embeddings for similarity tasks.



# Lecture 9: Text Generation with Transformers
*Instructor: Codruț-Georgian ARTENE*

---

## Lecture Overview
-   Brief History of Text Generation
-   Transformer-Based Text Generation
-   Decoding Strategies
-   Evaluation of Text Generation
-   Challenges and Future Directions
-   Summary and Q&A

---

## Lecture Context

### Recap of Prior Knowledge
-   **Word Embeddings:** Representing words as dense vectors.
-   **Recurrent Neural Networks (RNNs):** Sequential models for processing text.
-   **Transformers:** The current state-of-the-art architecture.

### Importance of Text Generation
Text generation is a cornerstone of modern AI, enabling applications such as:
-   Chatbots & Virtual Assistants
-   Language Translation
-   Text Summarization
-   Creative Writing (poetry, scripts)
-   Code Generation

---

## Brief History of Text Generation

### The Pre-Transformer Era

-   **Rule-Based Systems (Early Days):** Used handcrafted rules and templates. Very rigid.
-   **Statistical Models:**
    -   **N-gram Models:** Predicted the next word based on the probability of it occurring after the previous `n-1` words.
        -   **Limitations:** Struggled with long-range dependencies, produced repetitive text, and tended to memorize training data.
    -   **Hidden Markov Models (HMMs):** A probabilistic model for sequences.
-   **Recurrent Neural Networks (RNNs):**
    -   Models like LSTMs and GRUs processed text sequentially, maintaining a hidden state to capture context.
    -   **Limitations:** Suffered from the vanishing gradient problem, struggled to learn long-range dependencies, and were computationally inefficient due to their sequential nature.

---

## Transition to Transformers

Transformers revolutionized sequence modeling and text generation.

### Key Advantages over RNNs
-   **Attention Mechanism:** Capable of capturing both short-range and long-range dependencies effectively by allowing every token to look at every other token.
-   **Parallel Processing:** Processes all tokens in a sequence simultaneously, leading to much faster training and inference compared to the sequential nature of RNNs.

### Key Components of the Transformer Architecture
-   **Self-Attention:** For capturing relationships within the input sequence.
-   **Multi-Head Attention:** For capturing diverse contextual relationships from different perspectives.
-   **Positional Encoding:** To provide the model with a sense of word order.
-   **Feed-Forward Networks:** For complex transformations on each token's representation.
-   **Residual Connections & Layer Normalization:** To aid in efficient training of deep networks.

---

## Transformer Decoder for Text Generation

For text generation, we primarily use the **decoder** part of the Transformer architecture. While the original Transformer had an encoder-decoder structure (for translation), many modern language models (like GPT) are **decoder-only**.

### Autoregressive Generation
-   Transformers generate text **autoregressively**, meaning they produce the output sequence one token at a time.
-   The prediction for the current token is conditioned on all the tokens that have been generated previously.
-   **Example:** To generate "time" in "Once upon a time", the model takes "Once upon a" as input.

### The Generation Process: A High-Level View
1.  **Input Representation:** Words (tokens) are converted into numerical representations.
2.  **Embeddings:** The input to the Transformer decoder consists of token embeddings.
3.  **Positional Encoding:** Since Transformers process words in parallel and have no inherent sense of order, positional encodings are added to the token embeddings to inform the model about the sequence order.

### Positional Encoding vs. Positional Embeddings
-   **Positional Encoding (Fixed/Pre-defined):** Generated by a fixed, deterministic function (e.g., sine and cosine waves of different frequencies). They are *not* learned.
    -   **Advantage:** Can generalize to sequence lengths not seen during training.
-   **Positional Embeddings (Learned):** A learnable embedding layer where each position in the sequence is assigned a unique vector that is learned during training.
    -   **Advantage:** The model can learn the optimal positional representations.
    -   **Disadvantage:** May not generalize well to sequences longer than those seen in training.

### Inside the Decoder Block
-   **Masked Self-Attention:** This is the critical component for autoregressive generation. It prevents the decoder from "cheating" by looking at future tokens. When predicting the token at position `i`, the model can only attend to tokens from position `1` to `i-1`.
-   **Multi-Head Attention:** Allows the model to capture diverse relationships and understand context from multiple perspectives. In short: more heads = more ways to understand the context.
-   **Stacking Blocks:** Stacking multiple Transformer blocks allows the model to build progressively deeper and more abstract representations of the text, leading to better performance.
-   **Output Layer:** The final output of the decoder stack is passed through a linear layer and a **softmax** function to produce a **probability distribution** over the entire vocabulary for the next token.

---

## From Probabilities to Tokens: Decoding Strategies

The model's final output is a probability distribution. A **decoding strategy** is the algorithm used to select a single token from this distribution. The choice of strategy significantly impacts the fluency, coherence, and creativity of the generated text.

### 1. Greedy Decoding
-   **Strategy:** At each step, simply select the token with the highest probability.
-   **Advantages:** Computationally efficient and fast.
-   **Disadvantages:** Tends to produce repetitive, predictable, and uncreative text. It can easily get stuck in loops.

### 2. Beam Search
-   **Strategy:** Instead of just tracking the single best token, it keeps track of the `k` most probable sequences (beams) at each step. `k` is the "beam width".
-   **Advantages:** Finds higher-probability sequences than greedy search, offering a better trade-off between quality and computational cost.
-   **Disadvantages:** Still deterministic and can lack diversity. Computationally expensive for large beam sizes.

### 3. Sampling-Based Decoding
To introduce more randomness and creativity, we can sample from the probability distribution.

-   **Temperature Sampling:**
    -   Modifies the probability distribution using a `temperature` parameter (`τ`) before sampling.
    -   **Low temperature (`τ < 1`):** Makes the distribution "sharper," increasing the probability of high-probability words (closer to greedy).
    -   **High temperature (`τ > 1`):** Makes the distribution "flatter," increasing the probability of lower-probability words (more randomness).

-   **Top-k Sampling:**
    -   **Strategy:** Restrict the sampling pool to the `k` most likely tokens. The probabilities of these `k` tokens are then renormalized, and a token is sampled from this smaller set.
    -   **Advantages:** Balances randomness and quality, avoiding nonsensical words.
    -   **Choosing `k`:** A smaller `k` leads to less diversity (more conservative text), while a larger `k` leads to more diversity.

-   **Top-p (Nucleus) Sampling:**
    -   **Strategy:** Select the smallest set of tokens whose cumulative probability exceeds a threshold `p` (e.g., `p=0.9`). Sampling is then performed from this "nucleus" of high-probability tokens.
    -   **Advantages:** More adaptive than top-k, as the size of the sampling pool changes dynamically based on the model's confidence. Often leads to more natural and fluent text.

### Choosing the Right Strategy
-   **Greedy/Low-Temp:** Good for factual, precise tasks like code generation or short product descriptions.
-   **Beam Search:** Good for tasks where fluency and accuracy are important but some diversity is needed, like machine translation or summarization.
-   **Sampling (High-Temp, Top-k, Top-p):** Best for creative tasks like writing poetry, fiction, or dialogue, where diversity and novelty are valued.

---

## Evaluation of Text Generation

Evaluating generated text is challenging because quality is subjective. Unlike classification, there is no single perfect metric.

-   **Automatic Metrics:**
    -   **BLEU:** Measures n-gram overlap with reference texts. Primarily used in machine translation.
    -   **ROUGE:** Similar to BLEU but focuses on recall. Primarily used in summarization.
    -   **Perplexity:** Measures how well a model predicts a sample of text. Lower is better, but it doesn't always correlate with human-perceived quality.
    -   **Other:** BERTScore, METEOR, Word Mover's Distance (WMD).

-   **Human Evaluation:**
    -   Automatic metrics often don't correlate well with human judgment.
    -   Ultimately, human evaluation is necessary to assess true quality based on criteria like **fluency, coherence, relevance, and interestingness**.

---

## Challenges and Ethical Considerations

### Current Challenges
-   Maintaining coherence over very long passages.
-   Controlling style and content precisely.
-   Avoiding repetition.
-   **Hallucinations:** Generating factually incorrect or nonsensical information.
-   Mitigating bias present in the training data.

### Ethical Considerations
-   **Misinformation and Fake News:** The potential to generate convincing but false content.
-   **Deepfakes & Plagiarism:** Malicious use and academic dishonesty.
-   **Toxicity:** Generation of harmful or discriminatory language.
-   **Legal Aspects:** Issues of data protection, intellectual property, and regulations like the EU AI Act.

---

## Conclusions

-   Transformers have significantly advanced text generation, enabling more coherent, fluent, and contextually relevant text.
-   Decoding strategies are crucial for determining the style and quality of the generated output, offering trade-offs between creativity and coherence.
-   Significant challenges remain, particularly around factual accuracy (hallucinations), control, and bias.
-   Ethical considerations are paramount, as the technology can be used for both beneficial and harmful purposes.





# Lecture 10: Question Answering (QA) & Machine Translation (MT)
*Instructor: Codruț-Georgian ARTENE*

---

## Lecture Overview
-   **Introduction to QA and MT**
-   **Question Answering (QA)**
    -   Types of QA: Extractive vs. Abstractive
    -   Extractive QA: Concepts & Methods
    -   End-to-End Extractive QA: BERT, RoBERTa, XLNet
    -   Generative (Abstractive) QA: Concepts & Models (GPT, T5, BART)
    -   Evaluating QA: SQuAD, MRQA, ROUGE/BLEU
-   **Machine Translation (MT)**
    -   Statistical MT (SMT): A Brief Look
    -   Neural MT (NMT): Seq2Seq & Attention
    -   Transformer-Based NMT: The Revolution
    -   Evaluating MT: BLEU, ROUGE
-   **Conclusion & Future Directions**

---

## What are QA and MT?

-   **Question Answering (QA):** Systems that automatically answer questions posed by humans in natural language.
    -   **Goal:** Provide precise and relevant answers, either by extracting a span of text or generating new text based on a given context.
    -   **Input:** A question (e.g., "Why is the sky blue?") and often a context document.
    -   **Output:** An answer (e.g., "Scattering of sunlight..." or "Blue light is scattered more...").

-   **Machine Translation (MT):** Systems that automatically translate text or speech from one natural language to another.
    -   **Goal:** Produce fluent and accurate translations.
    -   **Input:** A source sentence (e.g., "Bonjour le monde").
    -   **Output:** A target sentence (e.g., "Hello world").

-   **Why study them?** They are core NLP tasks with huge real-world impact in search, virtual assistants, summarization, and global communication.

---

## Part 1: Question Answering (QA)

### The QA Landscape: Extract or Generate?

-   **Extractive QA:**
    -   Selects the answer as a **continuous span of text** directly from the provided context.
    -   **Assumption:** The answer exists verbatim within the text.
    -   *Example:* For "Is there a subscription fee for ChatGPT Plus?", the model extracts "$20".

-   **Abstractive (Generative) QA:**
    -   **Generates** the answer text, potentially paraphrasing, summarizing, or synthesizing information from the context.
    -   **Assumption:** The answer does not need to be a direct span. This allows for more natural, concise, or inferential answers.
    -   *Example:* For the same question, the model generates "Yes, there is a $20 subscription fee...".

-   **Other Dimensions:** QA systems can also be categorized as Open-Domain vs. Closed-Domain, Factoid vs. Conversational, etc.

### Extractive QA: Finding the Answer Span

-   **Task:** Given a question (Q) and a context passage (C), identify the continuous span of text (A) within C that answers Q.
-   **Early Approaches (Multi-stage Pipelines):**
    1.  **Question Processing:** Identify keywords and question type (Who, What, etc.).
    2.  **Document Retrieval:** Find relevant passages from a large corpus using methods like TF-IDF or BM25.
    3.  **Answer Extraction:** Use hand-crafted rules, patterns, or word overlap scores to find the answer.
    -   **Limitations:** Rules were brittle, had difficulty capturing semantics, and generalized poorly.

### Neural Networks for Extractive QA

-   **Key Idea:** Learn representations (embeddings) for the question and context, and then predict the **start and end positions** of the answer span within the context.
-   **Model Input:** Typically, the question and context are concatenated into a single sequence: `[CLS] Question [SEP] Context [SEP]`.
-   **Model Output:** Two probability distributions over the context tokens—one for the start index and one for the end index.
-   **Advantage:** This end-to-end approach learns semantic similarity and relevance automatically, making it far more robust than rule-based systems.

### End-to-End Extractive QA with Transformers

The revolution in extractive QA came from large-scale pre-trained language models like BERT.

-   **Mechanism:** A pre-trained model (which already understands language) is **fine-tuned** on a specific QA dataset (like SQuAD).
-   **Benefit:** This transfers the vast knowledge learned during pre-training to the specific task of QA.
-   **Models:** BERT, RoBERTa, XLNet.

#### BERT for Extractive QA
-   **Architecture:** Transformer-based encoder.
-   **Fine-tuning Process:**
    1.  **Input:** Concatenate the question and context: `[CLS] Q1 Q2 ... [SEP] C1 C2 ... [SEP]`.
    2.  **Output Layer:** Add two separate linear layers on top of BERT's final hidden states. Each layer is followed by a softmax.
        -   One layer predicts the probability of each context token being the **start** of the answer (`P_start`).
        -   The other layer predicts the probability of each token being the **end** of the answer (`P_end`).
    3.  **Training:** Minimize the cross-entropy loss for the true start and end positions.
    4.  **Inference:** Find the span `(start_i, end_j)` with `j >= i` that maximizes the score `P_start(i) * P_end(j)`.

#### RoBERTa for QA
-   **Improvements over BERT:** Trained longer on more data, removed the NSP task, and used dynamic masking.
-   **Result:** Often outperforms BERT on extractive QA benchmarks like SQuAD.

#### XLNet for QA
-   **Motivation:** Combines the strengths of autoregressive models (like GPT) and autoencoding models (like BERT) using **Permutation Language Modeling**. This learns a bidirectional context without needing a `[MASK]` token.
-   **Architecture:** Uses Transformer-XL, which incorporates recurrence to handle longer sequences.

### Abstractive QA: Generating the Answer

-   **Task Definition:** Given a question (Q) and context (C), generate a natural language answer (A). The answer is generated, not extracted.
-   **Advantage:** Can handle questions requiring inference, summarization, or synthesis, producing more natural responses.
-   **Architectures:**
    -   **Encoder-Decoder:** The encoder reads the question and context, and the decoder generates the answer. (e.g., T5, BART).
    -   **Decoder-only:** The model receives the question and context as a single prompt and generates the answer autoregressively. (e.g., GPT family).

#### Models for Abstractive QA

-   **GPT (Decoder-only):**
    -   Uses powerful generative capabilities, excelling at zero-shot and few-shot learning via prompting.
    -   **Weakness:** Can be more prone to hallucination if not prompted carefully, as it's not explicitly conditioned on context via an encoder.
-   **T5 (Text-to-Text Transfer Transformer):**
    -   An **encoder-decoder** model that treats every NLP task as a text-to-text problem.
    -   **QA Fine-tuning:** Input is formatted as `"question: <Q> context: <C>"`. The model learns to generate the target answer `<A>`.
    -   **Strength:** Versatile and performs well when fine-tuned.
-   **BART (Bidirectional and Auto-Regressive Transformer):**
    -   An **encoder-decoder** model where the encoder is bidirectional (like BERT) and the decoder is autoregressive (like GPT).
    -   **Strength:** Its denoising pre-training objective is well-suited for generation tasks that require understanding corrupted or incomplete input.

### Challenges in Abstractive QA
-   **Faithfulness / Hallucination:** Ensuring generated answers are supported by the context and not factually incorrect.
-   **Evaluation:** Evaluating generated text is hard. Metrics like ROUGE and BLEU have limitations; human evaluation is often necessary.
-   **Conciseness vs. Completeness:** Generating informative but not overly verbose answers.
-   **Prompt Engineering:** Performance is highly sensitive to the prompt structure.

### Evaluating QA Systems
-   **Datasets:**
    -   **SQuAD:** Primarily for Extractive QA.
    -   **Natural Questions (NQ):** Contains both short (extractive) and long (abstractive) answers.
    -   **ELI5 ("Explain Like I'm 5"):** Requires long-form, abstractive answers.
-   **Metrics:**
    -   **Extractive:** Exact Match (EM) and F1 Score.
    -   **Abstractive:** ROUGE (recall-focused), BLEU (precision-focused), and more recent embedding-based metrics like BERTScore.

---

## Part 2: Machine Translation (MT)

### The Goal
-   **Task:** Automatically convert a sequence of text from a source language to a target language.
-   **Challenges:** Lexical ambiguity, syntactic differences (word order), morphological complexity, idioms, and cultural nuances.

### Statistical Machine Translation (SMT) (~1990s-2014)
-   **Core Idea (Noisy Channel Model):** Find the target sentence `(e)` that maximizes the probability `P(e|f)`, which is proportional to `P(f|e) * P(e)`.
    -   `P(f|e)`: The **Translation Model**. How well do the phrases translate?
    -   `P(e)`: The **Language Model**. Is the output fluent in the target language?
-   **Phrase-Based SMT:** An improvement that translated contiguous sequences of words (phrases) instead of single words. It involved complex pipelines for word alignment, phrase extraction, and decoding.
-   **Limitations:** Complex components, disfluent output ("Translationese"), poor handling of long-range context, and resource-intensive.

### Neural Machine Translation (NMT)
-   **Core Idea:** Use a single, large neural network to map a source sentence directly to a target sentence.
-   **Architecture:** Typically an Encoder-Decoder framework.

#### Early NMT: RNN-based Seq2Seq
-   **Implementation:** Used RNNs (LSTMs/GRUs) for both the encoder and decoder.
-   **Bottleneck Problem:** Compressing the entire meaning of a long source sentence into one fixed-size context vector was very difficult.

#### The Attention Solution
-   **Mechanism (Bahdanau et al., 2014):** Allowed the decoder to "look back" at **all** hidden states of the encoder at each decoding step.
-   The decoder learns to dynamically place "attention" on the most relevant parts of the source sentence when generating each target word. This solved the bottleneck problem and dramatically improved performance, especially for long sentences.

### Transformer for NMT: Encoder-Decoder in Action
-   **Encoder Role (Source Understanding):**
    -   Processes the source sentence using self-attention to build context-aware representations.
    -   **Output:** A sequence of contextualized vectors (Keys `K_enc` and Values `V_enc`) for the entire source sentence.
-   **Decoder Role (Target Generation):**
    -   **Input:** Previously generated target tokens.
    -   **Process:** Each decoder layer has three key steps:
        1.  **Masked Self-Attention:** Attends to previously generated *target* tokens.
        2.  **Cross-Attention:** The core translation step. It uses its current state as a Query (`Q_dec`) to attend to the Keys (`K_enc`) and Values (`V_enc`) from the encoder. This is where the source directly influences the target generation.
        3.  **FFN:** Processes the information.
-   **Decoder-Only Transformers for NMT:** Large decoder-only models can also perform NMT through prompting (zero-shot or few-shot learning), but may be less robust than dedicated fine-tuned encoder-decoder models.

### How Good is the Translation? MT Evaluation
-   **Human Evaluation:** The gold standard for fluency and adequacy, but slow and expensive.
-   **Automatic Metrics:**
    -   **BLEU (Bilingual Evaluation Understudy):** A precision-focused metric that measures n-gram overlap with one or more human reference translations.
    -   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** A recall-focused metric.

### Challenges in MT
-   Low-Resource Languages
-   Domain Adaptation (e.g., medical, legal)
-   Handling Ambiguity and Nuance
-   Improving Evaluation Metrics
-   Computational Cost

---

## Summary: QA & MT
-   **Question Answering:**
    -   **Extractive (BERT, RoBERTa):** Find answer spans. Evaluated by EM/F1 on datasets like SQuAD.
    -   **Abstractive (GPT, T5, BART):** Generate answers. Evaluated by ROUGE/BLEU. Requires careful handling of faithfulness.
-   **Machine Translation:**
    -   Shifted from complex SMT pipelines to end-to-end NMT.
    -   Transformer architecture (Encoder-Decoder with Cross-Attention) dominates.
-   **Common Themes:** Transformer blocks and attention mechanisms are crucial across both advanced QA and MT.
-   **Ongoing Challenges:** Robustness, generalization, low-resource scenarios, evaluation, and computational cost remain active areas of research for both fields.




# Lecture 11: Part-of-Speech (POS) Tagging & Named Entity Recognition (NER)
*Instructor: Codruț-Georgian ARTENE*

---

## 1. Introduction to Part-of-Speech Tagging

### What is Part-of-Speech (POS) Tagging?
-   **Definition:** POS tagging is the process of assigning a grammatical category (part of speech) to each word in a text.
-   **Examples of Tags:** Noun, Verb, Adjective, Adverb, Pronoun, Preposition, etc.
-   **Task Format:**
    -   **Input:** A sequence of words (a sentence).
    -   **Output:** A sequence of tags, one for each word.
-   **Example:**
    -   **Sentence:** "The cat sat on the mat."
    -   **Tagged Output:** `The/DT cat/NN sat/VBD on/IN the/DT mat/NN ./ .`

### Why is POS Tagging Important?
POS tagging is a foundational task in NLP. The tags are crucial features for many higher-level tasks, including:
-   Parsing (syntactic analysis)
-   Named Entity Recognition (NER)
-   Information Extraction (IE)
-   Machine Translation (MT)
-   Sentiment Analysis

### POS Tagsets
A **tagset** is a predefined collection of POS tags. The choice of tagset depends on the language and the desired level of detail.

-   **Penn Treebank (PTB) Tagset:**
    -   Widely used for English.
    -   Contains around 36-45 main tags.
    -   **Examples:**
        -   `NN`: Noun, singular or mass (e.g., *cat*, *tree*)
        -   `NNS`: Noun, plural (e.g., *cats*, *trees*)
        -   `VB`: Verb, base form (e.g., *take*)
        -   `VBD`: Verb, past tense (e.g., *took*)
        -   `JJ`: Adjective (e.g., *big*)
        -   `DT`: Determiner (e.g., *the*, *a*)

-   **Universal Dependencies (UD) Tagset:**
    -   Aims for cross-linguistic consistency.
    -   Defines a smaller, universal set of 17 core tags (e.g., `NOUN`, `VERB`, `ADJ`, `ADV`).
    -   Often supplemented with language-specific features.

### Key Challenge: Ambiguity
The primary challenge in POS tagging is ambiguity. Many words can have multiple POS tags depending on their context. The goal of a POS tagger is to resolve these ambiguities.
-   "I **book** a flight." (`VB` - Verb)
-   "I read a **book**." (`NN` - Noun)
-   "Time **flies** like an arrow." (Is *flies* a plural noun `NNS` or a verb `VBZ`?)

---

## 2. Approaches to POS Tagging

1.  **Rule-Based Taggers:**
    -   Use hand-crafted rules based on linguistic knowledge.
    -   *Example Rule:* "If a word ends in '-ing' and is preceded by a form of 'to be', it’s a present participle (VBG)."
    -   **Pros:** Interpretable; can be accurate if rules are comprehensive.
    -   **Cons:** Brittle, difficult to create and maintain, not robust to new words.

2.  **Stochastic/Probabilistic Taggers:**
    -   Learn statistical models from large annotated corpora.
    -   **Examples:** Hidden Markov Models (HMMs), Conditional Random Fields (CRFs).
    -   **Pros:** Data-driven, handle ambiguity well, generally robust.
    -   **Cons:** Require large amounts of labeled training data.

3.  **Neural Network-Based Taggers:**
    -   Use neural architectures like RNNs, LSTMs, and Transformers.
    -   Often achieve state-of-the-art performance by learning complex features automatically.
    -   **Pros:** High accuracy; can leverage powerful pre-trained word embeddings.
    -   **Cons:** Can be "black boxes"; require significant computational resources.

---

## 3. Advanced POS Tagging Topics

### Evaluation of POS Taggers
-   Taggers are typically evaluated on a held-out test set with gold-standard tags.
-   **Accuracy** is the primary metric (percentage of correctly tagged tokens).
-   Accuracy is often reported separately for:
    -   **Known words:** Words seen in the training data.
    -   **Unknown words (OOV - Out-Of-Vocabulary):** Words not seen in training. Performance on OOV words is a key differentiator between models.

### Error Analysis
-   Analyzing which tags are most often confused (e.g., `NN` vs. `JJ`).
-   Identifying which types of words cause the most errors (ambiguous words, rare words).
-   This helps guide improvements, and confusion matrices are a useful tool for visualization.

### Handling Unknown Words (OOV)
This is a major challenge. Taggers use several clues:
-   **Morphological Clues:** Using prefixes and suffixes (e.g., `-ly` suggests an adverb, `-tion` suggests a noun) and capitalization (e.g., initial cap suggests a proper noun `NNP`).
-   **Contextual Clues:** The surrounding words provide strong evidence (e.g., in "a \_\_\_ book", the unknown word is likely an adjective `JJ`).
-   **Word/Character Embeddings (Neural Models):** Character-level models (CharCNNs, CharLSTMs) can create representations for OOV words based on their spelling. Subword tokenization (BPE, WordPiece) can represent OOV words as a sequence of known subwords.
-   **Backoff Strategies:** If a word is unknown, assign a default tag (like `NN`) or use a simpler OOV-specific model.

---

## 4. Introduction to Named Entity Recognition (NER)

### What is Named Entity Recognition (NER)?
-   **Definition:** NER is the task of identifying and classifying named entities (NEs) in text into predefined categories.
-   **Named Entities:** Mentions of real-world objects like persons, locations, organizations, dates, monetary values, etc.
-   **Example:**
    -   **Sentence:** "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976."
    -   **NER Output:**
        -   Apple Inc. `[ORG]`
        -   Steve Jobs `[PER]`
        -   Cupertino `[LOC]`
        -   April 1, 1976 `[DATE]`
-   NER is a subtask of **Information Extraction (IE)** and is crucial for understanding "who did what to whom, where, and when."

### Common Named Entity Types
The set of types varies by application and dataset.
-   **Core Types (e.g., CoNLL-2003):**
    -   `PER`: Person (e.g., *Jeff Bezos*)
    -   `LOC`: Location (e.g., *Paris*)
    -   `ORG`: Organization (e.g., *Google*)
    -   `MISC`: Miscellaneous (nationalities, events, products)
-   **Numeric/Temporal Types:** `DATE`, `TIME`, `MONEY`, `PERCENT`.
-   **Domain-Specific Types:**
    -   **Biomedical:** *Gene, Protein, Disease*
    -   **Financial:** *Stock Symbol, Company Ticker*
-   **Granularity:** NER can be coarse-grained (few types) or fine-grained (many specific types, e.g., distinguishing between `CITY`, `STATE`, `COUNTRY` under the `LOC` type).

### NER as a Sequence Labeling Task
NER is typically framed as a sequence labeling problem, similar to POS tagging.
-   **IOB / BIO Tagging Scheme:** The most common scheme.
    -   `B-TYPE`: **B**eginning of a named entity of TYPE.
    -   `I-TYPE`: **I**nside a named entity of TYPE.
    -   `O`: **O**utside any named entity.
-   **Example:** `Steve [B-PER] Jobs [I-PER] founded [O] Apple [B-ORG] Inc. [I-ORG] ...`
-   **Challenges in NER:**
    -   **Boundary Detection:** Identifying the exact start and end of an NE (e.g., "Bank of America" vs. "Bank").
    -   **Ambiguity:** "Washington" (PER or LOC?), "May" (DATE or PER?).
    -   **Context Dependence:** "Ford" can be a person, organization, or location.
    -   **OOV Entities:** New entities not seen in training.

---

## 5. Approaches to NER

### Rule-Based NER Systems
-   Rely on hand-crafted rules, linguistic patterns, and orthography.
-   **Components:**
    -   **Gazetteers (Dictionaries):** Lists of known entities (cities, countries, etc.).
    -   **Regular Expressions:** Patterns for specific entity types (e.g., dates, monetary values).
    -   **Syntactic/Semantic Constraints:** E.g., "If a capitalized sequence follows 'Mr.', it’s likely a PER."
-   **Pros:** Interpretable, accurate for well-defined types, no annotated training data needed.
-   **Cons:** Labor-intensive, brittle, low recall, hard to handle ambiguity.

### Statistical NER: HMMs and CRFs
-   **HMMs for NER:**
    -   **States:** The NER tags (B-PER, I-PER, etc.).
    -   **Observations:** The words.
    -   Learns transition probabilities (e.g., `P(I-PER|B-PER)`) and emission probabilities (e.g., `P("Paris"|B-LOC)`) from data.
-   **CRFs for NER:**
    -   Became the dominant approach before deep learning.
    -   Directly models `P(Tags | Words)`, allowing for a rich set of features.
    -   **Feature Engineering is Key:** Features include the current word, surrounding words, prefixes/suffixes, capitalization, POS tags, gazetteer lookups, etc.

### Neural NER: BiLSTMs and Transformers

#### Bidirectional LSTMs (BiLSTMs) for NER
-   Context from both the past (left) and future (right) is crucial for NER.
-   A **BiLSTM** consists of two LSTMs: one processes the sequence left-to-right, the other right-to-left. Their hidden states are concatenated at each time step to form a rich, bidirectional representation of each word.

#### BiLSTM-CRF Architecture
-   This became the state-of-the-art model for several years.
-   **Limitation of simple BiLSTM:** A softmax output makes independent tagging decisions for each token, which can lead to invalid tag sequences (e.g., `O B-PER I-LOC`).
-   **Combining Strengths:**
    1.  The **BiLSTM** layer is good at feature extraction, producing powerful contextual scores (potentials) for each tag at each position.
    2.  A **CRF** layer is added on top. It is good at modeling output dependencies, learning constraints between adjacent tags (e.g., an `I-PER` tag must follow a `B-PER` or another `I-PER`).
    3.  The CRF layer finds the optimal sequence of tags by considering both the BiLSTM's emission scores and the learned tag transition probabilities.
-   **Inference:** The **Viterbi algorithm** is used to find the most likely tag sequence.

#### BERT for NER (Fine-tuning)
-   Pre-trained BERT can be fine-tuned for NER by adding a small task-specific classification layer.
-   **Architecture:**
    1.  Pass the input sequence through the pre-trained BERT encoder to get contextualized hidden state vectors for each token.
    2.  For each token's final hidden state, add a single linear layer plus a softmax to classify it into one of the NER tags.
-   **Handling WordPiece Tokenization:** Since WordPiece splits words (e.g., "recognizing" → "recogn", "##izing"), a common strategy is to use the prediction for the first subword token as the tag for the entire original word.
-   **BERT + CRF Layer:** Similar to BiLSTM-CRF, a CRF layer can be added on top of BERT's output to model tag dependencies, which often yields slight improvements.

---

## 6. Advanced NER Topics

### Framing NER as Question Answering
-   **Idea (MRC for NER):** Reformulate NER as an Extractive Question Answering task.
-   **Process:**
    1.  For each entity type, formulate a natural language question (e.g., For `PER`, ask "Who is the person in this sentence?").
    2.  Feed the question and the context sentence to a QA model (like BERT).
    3.  The model's task is to find the "answer" (the entity span) in the context.
-   **Advantages:** Can handle nested entities, allows for zero/few-shot learning for new entity types, and is more interpretable.
-   **Disadvantages:** Slower inference (one pass per entity type), performance is sensitive to question phrasing.

### Nested Named Entities
-   Entities that are part of other entities (e.g., "[University of [California, [Berkeley]]]").
-   Standard BIO tagging cannot represent nested structures.
-   **Approaches:**
    -   **Cascaded Models:** Run multiple NER systems in layers.
    -   **Span-based Models:** Enumerate all possible spans in a sentence and classify each one.
    -   **Sequence-to-Sequence Models:** Frame NER as a generation task.

### Fine-grained Named Entity Recognition
-   Aims to identify entities with much more specific types than standard NER (e.g., `ATHLETE`, `POLITICIAN`, `ARTIST` instead of just `PER`).
-   **Challenges:** Large number of types, data sparsity, hierarchical types, and ambiguity.

### Other Advanced Topics
-   **Cross-lingual NER:** Applying NER models trained on one language (e.g., English) to another (e.g., a low-resource language).
-   **Low-Resource NER:** Developing NER systems with very limited annotated data.
-   **Entity Linking (EL/NED):** After identifying an entity, linking it to a unique entry in a knowledge base (e.g., Wikipedia).
-   **Open-Domain NER:** Identifying entities without a predefined set of types.

---

## 7. Applications of POS Tagging and NER

POS and NER are fundamental components in many downstream applications.

-   **Information Extraction (IE):** NER identifies the key entities (who, what, where), and POS provides syntactic information to understand the relationships between them.
-   **Natural Language Understanding (NLU):**
    -   **Syntactic Parsing:** POS tags are essential input.
    -   **Semantic Role Labeling (SRL):** NER helps identify arguments, POS helps identify predicates (verbs).
    -   **Coreference Resolution:** NER provides the initial entity mentions.
    -   **Question Answering (QA):** Helps identify the entity type being asked for in the question and locate candidate answers in documents.
-   **Other Applications:** Machine Translation, Sentiment Analysis, Text Summarization, Dialogue Systems, Content Recommendation, and more.

---

## 8. Recent Trends and Future Directions

### Large Language Models (LLMs) for POS & NER
-   **Zero-Shot / Few-Shot Learning:** LLMs can perform NER and POS tagging by formulating the task in a textual prompt, often with just a few examples (**In-Context Learning**).
-   **Pros:** Reduced need for large annotated datasets; flexibility to define new types on the fly.
-   **Cons:** High computational cost; issues with controllability and consistency; "black box" nature; potential for hallucination; data privacy concerns. Fine-tuning smaller, specialized models like BERT-CRF can still be more efficient and robust for well-defined tasks.

### Other Trends
-   **Multimodal NER/POS:** Incorporating information from images or audio.
-   **Robustness and Generalization:** Improving performance on noisy text.
-   **Explainability:** Understanding why models make certain decisions.
-   **Ethical Considerations & Bias Mitigation:** Addressing and reducing biases in models.
-   **Efficient Models:** Developing smaller, faster models via knowledge distillation or quantization.

---

## 9. Conclusion

-   **POS tagging** and **NER** are fundamental NLP tasks that provide crucial lexical and semantic information.
-   **Evolution of Models:** The field has evolved from rule-based systems to statistical models (HMMs, CRFs), then to neural architectures (BiLSTM-CRF), and now to powerful pre-trained Transformers (BERT) and LLMs.
-   **Key Challenges Persist:** Ambiguity, OOV words, domain adaptation, and handling complex structures like nested entities remain active areas of research.
-   **Wide Range of Applications:** These tasks are critical enablers for almost all higher-level NLP applications.





# Lecture 12 - Reasoning in NLP & AI Agents
*Instructor: Codruț-Georgian ARTENE*

---

## 1. Introduction & The Need for Reasoning

### Beyond Pattern Matching: Why Reasoning is Crucial
Modern Large Language Models (LLMs) excel at tasks like text generation, translation, and summarization. These capabilities are based on learning statistical patterns from vast amounts of data. However, they often struggle with tasks that require a deeper level of understanding, such as:
-   Logical inference
-   Planning
-   Dealing with novel, unseen situations

**Examples:**
-   **Pattern Matching:** *"The capital of France is Paris."* (A learned fact).
-   **Reasoning (Deduction):** *"If all birds can fly, and a robin is a bird, can a robin fly?"* (Requires applying a rule).
-   **Reasoning (Inference):** *"Why did character X do that?"* (Requires inferring intent or causality).

**Limitations of purely data-driven approaches:**
-   **Lack of transparency (black box):** It's hard to know *how* an LLM arrived at an answer.
-   **Difficulty with complex logic:** Negation, disjunction, and other logical structures can be challenging.
-   **Prone to "hallucinations":** Generating factually incorrect but fluent-sounding text.
-   **Poor out-of-distribution performance:** Struggles to generalize beyond observed patterns.

Reasoning is the key to providing **robustness, explainability, and the ability to handle novelty.**

### Defining "Reasoning" in Modern AI
Reasoning is the process of drawing conclusions or making inferences from available information. In the context of NLP and AI Agents, this includes several facets:
-   **Logical Inference:** Strictly deriving conclusions from premises (deduction).
-   **Causal Reasoning:** Understanding cause-and-effect.
-   **Temporal Reasoning:** Understanding sequences of events over time.
-   **Spatial Reasoning:** Understanding locations and spatial relations.
-   **Abductive Reasoning:** Inferring the *best explanation* for an observation.
-   **Planning:** Devising a sequence of actions to achieve a goal.
-   **Commonsense Reasoning:** Using intuitive knowledge about the world.

### LLMs and Transformers: Potential Reasoning Engines?
-   LLMs have shown **emergent abilities** on tasks that seem to require reasoning (e.g., arithmetic, multi-hop QA).
-   **Hypothesis:** These reasoning capabilities are implicitly encoded in the model's vast parameters and attention mechanisms.
-   **Opportunity:** We can leverage the powerful language capabilities of LLMs for reasoning tasks.
-   **Challenges:**
    -   Lack of explicit, verifiable reasoning steps.
    -   Difficulty with systematic, compositional reasoning.
    -   Inconsistent performance, especially on novel problems.
    -   How do we reliably control or extract the "reasoning" from the black box?

---

## 2. Bridging Symbolic Logic and Neural Models

### Recap: Core Concepts of Formal Logic
-   **Propositional Logic (PL):** Deals with propositions (declarative sentences) and logical connectives (¬, ∧, ∨, →, ↔). It has formal rules for syntax, semantics (truth values), and inference (e.g., Modus Ponens).
-   **First-Order Logic (FOL):** Extends PL to include predicates, variables, constants, and quantifiers (∀, ∃), allowing for reasoning about objects and their properties.

Formal logic provides a precise, unambiguous framework for representing knowledge and performing sound inference.

### Symbolic Approaches to NLP Reasoning
Historically, AI research focused on symbolic systems.
-   **Rule-based Systems:** Knowledge is represented as IF-THEN rules, processed by an inference engine. Used in early Expert Systems.
-   **Theorem Provers:** Prove theorems from a set of axioms using logical inference rules. Can be used for logical QA.
-   **Strengths:** Transparency, explainability, guarantees of soundness.
-   **Weaknesses:** Brittleness (fail on unseen data), knowledge acquisition bottleneck (hard to write all rules), difficulty handling uncertainty.

### The Gap Between Symbolic and Neural
-   **Symbolic systems** operate on discrete symbols with clear semantics.
-   **Neural networks** operate on continuous vectors where meaning is distributed and emergent.
-   **The Challenge:** How can a continuous vector capture discrete logical properties? How do neural nets compose meaning systematically like logical formulas?

### Neuro-Symbolic Approaches
-   **Goal:** Combine the learning power of neural networks with the reasoning capabilities and transparency of symbolic methods.
-   **Motivations:** Improve robustness, incorporate prior knowledge, gain explainability, and reduce data requirements.
-   **Architectures:**
    -   **Neural-Guided Symbolic:** A neural network predicts parameters or guides the search in a symbolic system (e.g., predicting logical forms).
    -   **Symbolic-Guided Neural:** Symbolic rules constrain or regularize the training of a neural network.
    -   **Integrated Systems:** Neural and symbolic components interact dynamically.

**Examples:**
-   **Semantic Parsing:** A neural network predicts a logical query (like SPARQL) from a natural language question.
-   **Multi-hop QA:** A neural network extracts facts, and a symbolic component (or a module simulating one) chains them together to find the answer.

---

## 3. Knowledge Representation for Reasoning

### The Need for Explicit Knowledge
While LLMs capture vast implicit knowledge in their parameters, this knowledge is not structured, easily accessible, or verifiable. **Explicit knowledge representations** provide structured, reliable information about the world.

This is essential for:
-   Reliable, step-by-step reasoning.
-   Answering specific factual questions.
-   Understanding complex relationships.
-   Providing explanations for model outputs.

### Key Representation Methods
-   **Ontologies:** A formal specification of a domain's concepts, properties, and interrelationships.
    -   Uses formal languages like OWL.
    -   Provides hierarchical structure (e.g., *is-a* relationships).
    -   Allows for formal reasoning to check consistency and infer new facts.
-   **Knowledge Graphs (KGs):** A graph-based representation where nodes are entities and edges are relationships.
    -   Often represented as `(Subject, Predicate, Object)` triples.
    -   Constructed from text using NLP tasks like NER and Relation Extraction.
    -   Provides factual background knowledge for QA, dialogue, and recommendation systems.
    -   Reasoning can involve path finding, rule-based inference, or embedding-based methods.
-   **Frames & Scripts:**
    -   **Frames (Minsky):** Data structures for stereotypical situations with slots for roles and properties (e.g., a *Restaurant Frame*).
    -   **Scripts (Schank & Abelson):** Frames that represent sequences of events (e.g., a *Restaurant Script* with steps like "entering," "ordering," "paying").
-   **Commonsense Knowledge Bases:** Attempt to explicitly capture the intuitive knowledge humans use daily.
    -   **ConceptNet:** A semantic network of commonsense concepts.
    -   **ATOMIC:** A knowledge graph of "if-then" inferential knowledge about social events.

### Integrating External Knowledge into LLMs
-   **Fine-tuning:** Fine-tune an LLM on tasks that require using an external KB.
-   **Prompting (RAG):** Include relevant knowledge snippets directly in the prompt.
-   **Architectural Modifications:** Design models with specific modules for accessing and reasoning over KBs (e.g., Graph Neural Networks).

---

## 4. Automated Reasoning Methods and LLMs

### Main Types of Reasoning
-   **Deduction:** From general rules to specific conclusions (truth-guaranteed).
-   **Induction:** From specific observations to general rules (probabilistic).
-   **Abduction:** Inferring the most likely explanation for an observation (best guess).

### How LLMs Handle Formal Reasoning
-   **Deductive Reasoning (e.g., Textual Entailment):** LLMs can achieve high accuracy on standard benchmarks but struggle with complex, multi-step deductions. They learn statistical patterns associated with logical relationships rather than performing formal logic.
-   **Inductive Reasoning:** This is the core of how LLMs are trained! They induce statistical patterns and world knowledge from massive corpora. However, explicitly learning and manipulating symbolic rules remains a challenge.
-   **Abductive Reasoning (e.g., Story Understanding):** LLMs can often generate plausible explanations for events, relying heavily on the implicit commonsense knowledge acquired during training.

### How LLMs Exhibit/Simulate Reasoning
-   The debate continues: Do LLMs *truly* reason, or do they just simulate it through sophisticated pattern matching?
-   **Evidence for Simulation:** Performance improves with scale; attention patterns can highlight relevant information.
-   **Limitations:** Brittleness, sensitivity to phrasing, and inability to handle novel logical structures suggest it is often simulation rather than robust, systematic reasoning.

### Techniques to Enhance LLM Reasoning
-   **Fine-tuning** on specific reasoning datasets.
-   **Incorporating External Tools** (calculators, theorem provers).
-   **Advanced Prompting Strategies** (like Chain-of-Thought).
-   **Neuro-Symbolic Integration**.

---

## 5. Commonsense Reasoning with LLMs

### The Challenge of Commonsense
-   Commonsense is the intuitive knowledge we use to understand the world. It is vast, diverse, implicit in language, and context-dependent.
-   It is a major challenge for AI because it's difficult to formalize and requires connecting language to real-world experiences.

### How LLMs Acquire Commonsense
-   LLMs are trained on massive text datasets where commonsense knowledge is implicitly embedded in statistical patterns.
-   They learn associations (e.g., "if someone is hungry, they might want to eat") through statistical learning, not explicit rule acquisition.
-   This implicit knowledge is powerful but can be fragile.

### Evaluating Commonsense in LLMs
-   Specialized benchmarks are needed to probe this capability.
-   **Examples:** CommonsenseQA, WinoGrande (pronoun resolution), PIQA (physical interactions).

### Techniques to Improve LLM Commonsense
-   **Incorporate KBs:** Use RAG to retrieve facts from KBs like ConceptNet or ATOMIC.
-   **Generate Explanations:** Prompt LLMs to explain *why* a commonsense inference is valid.
-   **Prompting:** Use CoT to guide the model through commonsense steps.

---

## 6. Advanced Prompting and Architectures for Reasoning

Basic prompting (zero-shot, few-shot) is often insufficient for complex reasoning.

-   **Chain-of-Thought (CoT) Prompting:** Elicits intermediate reasoning steps from the LLM before the final answer, guiding its internal process.
-   **Tree-of-Thought (ToT):** Explores multiple reasoning paths simultaneously, forming a tree of "thoughts" that can be evaluated.
-   **Graph-of-Thought (GoT):** A more general approach allowing for arbitrary connections between thoughts, better for complex, non-linear dependencies.
-   **Retrieval-Augmented Generation (RAG):** Augments the LLM with an external knowledge source to provide a factual basis for reasoning and reduce hallucinations.
-   **Memory Mechanisms:** External memory (vector databases, KGs) is crucial for agents to remember past information beyond the limited context window.

---

## 7. LLM-based Agents and Reasoning

### What is an LLM-based Agent?
An AI system that **perceives** its environment, **reasons** to make decisions, and **takes actions** to achieve goals. LLMs are increasingly used as the "brain" or core reasoning component.

-   **Typical Architecture:**
    1.  **Perception:** Receives input (text, observations).
    2.  **Reasoning/Planning:** The LLM processes input and decides on the next action.
    3.  **Action:** Interacts with the environment (generates text, calls tools).
    4.  **Memory:** Stores past experiences, knowledge, and goals.

### Agent Capabilities Driven by LLM Reasoning
-   **Instruction Following:** Interpreting natural language commands.
-   **State Tracking:** Maintaining an understanding of the environment.
-   **Goal Interpretation:** Decomposing high-level goals into sub-goals.
-   **Planning:** Devising a sequence of actions. This can be done by prompting the LLM directly or by integrating it with symbolic planners.
-   **Tool Use / Function Calling:** Determining when and how to call external tools (APIs, calculators, web search) to extend its capabilities.
-   **Self-Reflection and Learning:** Critiquing its own plans and learning from feedback to improve over time.

---

## 8. Applications and Future Directions

### Applications of Reasoning
-   **Advanced QA:** Multi-hop, temporal, and counterfactual QA.
-   **Dialogue Systems:** Tracking state, recognizing intent, and generating consistent, relevant responses.
-   **Information Extraction:** Inferring implicit relationships and ensuring consistency when building KBs.
-   **Creative Tasks:** Maintaining plot consistency, temporal coherence, and causal links in story generation.

### Open Challenges and Future Directions
-   **Robustness:** Making reasoning less brittle.
-   **Transparency & Explainability:** Understanding and verifying the reasoning process.
-   **Compositional Generalization:** Systematically combining learned concepts to solve novel problems.
-   **Scaling Neuro-Symbolic AI:** Making integrated approaches practical for large-scale use.
-   **Evaluating True Reasoning:** Developing benchmarks that can distinguish genuine reasoning from sophisticated pattern matching.




# Lecture 13 - NLP in Practice and Evaluation of NLP Models
*Adapting, Optimizing, Evaluating, and Understanding Modern NLP Systems*
*Instructor: Codruț-Georgian ARTENE*

---

## Course Roadmap
1.  **Foundations & NLP Models Recap**
    -   NLP in Practice
    -   Review of key NLP Models
2.  **NLP Models Training and Adaptation**
    -   Pre-training
    -   Transfer Learning & Fine-tuning
3.  **Efficient NLP: Making Models Practical**
    -   Quantization, Knowledge Distillation, Inference Optimization
4.  **NLP Models Evaluation**
    -   Standard Evaluation Metrics for NLU & NLG
5.  **Bias & Fairness**
6.  **Human Evaluation, Interpretability & Ethics**
7.  **Practical Considerations & Future Trends**
8.  **Conclusion**

---

## 1. Foundations & NLP Models Recap

### Why NLP in Practice Matters
Understanding the underlying models and processes is crucial for:
-   Effective application development
-   Informed model selection
-   Addressing limitations and ethical concerns
-   Pushing the boundaries of research and innovation
This lecture aims to bridge the gap between **theory and practical implementation**.

### From Theory to Practice: Key Challenges
-   **Computational cost** (Training & Inference)
-   **Memory footprint**
-   **Latency requirements**
-   **Limited labeled data** for specific tasks
-   **Deployment constraints** (Edge devices, cloud costs)
-   Ensuring **reliability, fairness, and robustness**

### Recap: The Evolution of NLP Models

#### Pre-Transformer Era (Sequential Processing)
-   **Core Idea:** Process text sequentially, word-by-word.
-   **Architectures:** Recurrent Neural Networks (RNNs), LSTMs, GRUs.
-   **Mechanism:** Maintain a hidden state that summarizes past information.
-   **Tasks:**
    -   **NLU (Sentiment, NER):** Use the final hidden state as input to a classifier.
    -   **NLG (Translation, Summarization):** Use a Seq2Seq architecture where an encoder RNN processes the input and a decoder RNN generates the output.
-   **Limitations:** Difficulty with long-range dependencies, vanishing/exploding gradients, information bottlenecks, and lack of parallelization.

#### The Modern NLP Stack: Transformer Era
-   **The Revolution:** Dominance of architectures like BERT, GPT, and T5.
-   **Pre-training Paradigm:** Models learn rich linguistic representations from massive text corpora using self-supervised objectives.
-   **Pre-trained Models (PTMs) / Foundation Models:**
    -   **Encoder-Only (BERT, RoBERTa):** Bidirectional context, strong for NLU tasks (classification, NER, extractive QA).
    -   **Decoder-Only (GPT series, Llama):** Autoregressive (causal), strong for NLG tasks (generation, summarization).
    -   **Encoder-Decoder (T5, BART):** Versatile for sequence-to-sequence tasks (translation, generative QA).
-   **Key Idea:** Leverage general knowledge from pre-training and adapt it to specific downstream tasks.

---

## 2. NLP Models Training and Adaptation

### Pre-training: Learning General Language Representation
-   **Goal:** Train a model on vast amounts of unlabeled text to learn grammar, facts, and context understanding.
-   **Method:** Self-supervised learning, where labels are derived from the input data itself.
-   **Key Objectives:**
    -   **Masked Language Modeling (MLM) (BERT-style):** Predict masked tokens from bidirectional context.
    -   **Causal Language Modeling (CLM) (GPT-style):** Predict the next token from unidirectional (left) context.
    -   **Sequence-to-Sequence Objectives (T5, BART-style):** Reconstruct original text from a corrupted version (e.g., text infilling/denoising).

### Transfer Learning in NLP
-   **Concept:** Leverage knowledge from a pre-trained model (source task) for a new, specific downstream application (target task).
-   **Why it Dominates:**
    -   **Data Efficiency:** Requires less labeled data for the target task.
    -   **Performance:** Often achieves state-of-the-art results.
    -   **Accessibility:** PTMs are readily available (e.g., Hugging Face Hub).
    -   **Reduced Cost:** Fine-tuning is much cheaper than pre-training from scratch.

### Fine-tuning: Adapting PTMs
-   **Goal:** Specialize a general PTM for a specific task.
-   **The Mechanics:**
    1.  **Select PTM:** Choose the appropriate architecture (Encoder, Decoder, etc.).
    2.  **Add/Replace Head:** Add a task-specific classification layer (the "head").
    3.  **Train on Labeled Data:** Continue training, usually with a small learning rate, to adjust weights for the new task.
    -   **Key:** Avoid **catastrophic forgetting** of the powerful pre-trained knowledge.

### Fine-tuning Strategies: Full vs. Parameter-Efficient (PEFT)
-   **Full Fine-tuning:**
    -   **Method:** Update all parameters of the PTM.
    -   **Pros:** Often the highest performance.
    -   **Cons:** High memory/storage cost (a full model copy per task), computationally expensive.
-   **Parameter-Efficient Fine-tuning (PEFT):**
    -   **Goal:** Adapt a model by modifying only a small subset of its parameters.
    -   **Benefits:** Reduced memory, storage, and training time; robust to forgetting; easy multi-task deployment.
    -   **Popular PEFT Methods:**
        -   **Adapters:** Insert small, trainable bottleneck modules between frozen Transformer layers.
        -   **LoRA (Low-Rank Adaptation):** Decompose the weight *update* matrix into two smaller, low-rank matrices and train only those. (No inference latency).
        -   **Prompt/Prefix Tuning:** Keep the PTM frozen and learn a small set of "soft prompt" vectors that are prepended to the input.

### Zero-shot & Few-shot Prompting
-   An alternative to fine-tuning, dominant with very large decoder-only models (GPT-3+).
-   The model performs a task by following instructions and/or examples provided directly in the prompt, **without any weight updates**.
-   **Zero-shot:** Provide only the task description.
-   **Few-shot (In-Context Learning):** Provide a few input-output examples to demonstrate the task.

---

## 3. Efficient NLP: Making Models Practical

### The Problem: Large Models vs. Real-world Constraints
Modern NLP models are huge, requiring massive compute, memory, and energy. This conflicts with real-world constraints like inference latency, throughput, and deployment cost. The goal of model compression is to make models smaller, faster, and cheaper without sacrificing too much accuracy.

### Overview of Model Compression & Optimization
-   **Parameter Reduction:**
    -   **Pruning:** Remove redundant/unimportant parameters (weights, neurons, heads). Can be *unstructured* (individual weights) or *structured* (entire structures).
    -   **Knowledge Distillation (KD):** Train a smaller "student" model to mimic the outputs (and internal representations) of a larger "teacher" model.
-   **Numerical Representation:**
    -   **Quantization:** Reduce the numerical precision of model weights and activations (e.g., from 32-bit float to 8-bit or 4-bit integer).
-   **Inference Optimization:**
    -   **KV Caching:** In generative models, reuse attention key/value pairs from previous steps to dramatically speed up token generation.
    -   **Compilers (ONNX, TensorRT):** Optimize the model's computation graph through techniques like layer fusion.

### A Closer Look at Quantization
-   **Concept:** Reduce the number of bits used per parameter (e.g., FP32 → INT8).
-   **Motivation:** Size reduction (4x for INT8), faster computation (integer math), lower energy consumption.
-   **Impact:** This mapping is **lossy**, as multiple FP32 values may map to the same integer value, introducing **quantization error**.
-   **Techniques:**
    -   **Post-Training Quantization (PTQ):** Quantize a model *after* it has been fully trained. Simple and fast, but can lead to significant accuracy degradation.
    -   **Quantization-Aware Training (QAT):** Simulate the effects of quantization *during* training, allowing the model to adapt and become more robust to precision loss. More complex, but yields better accuracy.

---

## 4. NLP Models Evaluation

Rigorous evaluation is essential to measure performance, compare models, understand weaknesses, and guide development.

### NLU Metrics
-   **Classification (Sentiment, Topic ID):**
    -   **Accuracy, Precision, Recall, F1-Score.**
    -   **AUC-ROC:** Measures discrimination ability.
    -   **Averages (for multi-class):** Macro, Micro, Weighted.
-   **Sequence Labeling (NER, POS Tagging):**
    -   **Entity-Level P/R/F1 (CoNLL standard):** Considers an entity correct only if both the **span** and **type** match the ground truth exactly.
-   **Extractive QA (SQuAD-style):**
    -   **Exact Match (EM):** Percentage of predictions that match the ground truth answer exactly.
    -   **F1-Score:** Measures the token overlap (bag-of-words) between the prediction and ground truth.

### NLG Metrics
Evaluating generated text is hard because there are many possible correct outputs. These metrics compare a system-generated text to human-written references.

-   **BLEU (Bilingual Evaluation Understudy):**
    -   Focuses on n-gram **precision**. Penalizes translations that are too short (Brevity Penalty). Primarily for machine translation.
-   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
    -   Focuses on n-gram **recall**. Primarily for text summarization.
    -   Variants include ROUGE-N (n-gram overlap) and ROUGE-L (Longest Common Subsequence).
-   **METEOR:**
    -   Balances precision and recall (F-score), but weights recall higher.
    -   Considers word order (via a chunk penalty) and semantic similarity (synonyms, stemming).
-   **Model-Based Metrics:**
    -   **BERTScore:** Computes cosine similarity between contextual embeddings (from BERT) of tokens in the candidate and reference texts. Correlates better with human judgment than n-gram overlap metrics.
-   **Intrinsic Metrics:**
    -   **Perplexity:** Measures a model's uncertainty in predicting text. Lower perplexity often correlates with higher fluency.

**Conclusion on Automatic Metrics:** Useful for development, but they often have weak correlation with human judgment and MUST be complemented by human evaluation.

---

## 5. Bias & Fairness

-   **Bias:** Systematic deviation reflecting societal stereotypes or imbalances from training data, leading to unfair outcomes for certain groups.
-   **Sources:** Data Bias (representation, stereotypes), Annotation Bias, Model Bias.
-   **Measuring Bias:**
    -   **Intrinsic (Embeddings):** Examine the model's internal representations (e.g., Word Embedding Association Test - WEAT).
    -   **Extrinsic (Downstream Task):** Evaluate how bias affects model performance on a specific task (e.g., performance disparities across demographic groups, counterfactual testing).
-   **Fairness Concepts:**
    -   **Group Fairness:** Parity across groups (e.g., Demographic Parity, Equalized Odds, Equal Opportunity).
    -   **Individual Fairness:** Similar individuals should be treated similarly.
-   **Mitigation Strategies:**
    -   **Pre-processing (Data):** Augmentation, re-sampling, data debiasing.
    -   **In-processing (Training):** Adversarial debiasing, adding fairness penalties to the loss function.
    -   **Post-processing (Outputs):** Adjusting prediction thresholds per group.

---

## 6. Human Evaluation, Interpretability & Ethics

### Limitations of Automatic Metrics & Need for Human Judgment
Automatic metrics fail to capture critical aspects of language, especially for NLG:
-   Fluency, Coherence, Consistency
-   Factual Accuracy / Hallucination
-   Appropriateness, Tone, Toxicity, Bias
**Conclusion:** Human evaluation is essential for a comprehensive assessment.

### Explainable AI (XAI) in NLP
Deep learning models are often "black boxes." We need XAI for:
-   **Trust, Debugging, Fairness Audits, and Compliance** (e.g., GDPR's "right to explanation").
-   **Techniques:**
    -   **Attention Visualization:** *Caution: Attention is not Explanation!* It shows information flow, not necessarily feature importance.
    -   **Gradient-based (Saliency):** Highlights input tokens that most affect the output score.
    -   **Occlusion/Perturbation:** Masking parts of the input to see how the output changes.
    -   **LIME/SHAP:** Model-agnostic methods that explain individual predictions.

### Broader Ethical Considerations
Beyond bias, other ethical issues include:
-   **Privacy:** Memorization and leakage of sensitive data.
-   **Misinformation:** Malicious use for fake news and propaganda.
-   **Dual Use:** Fake reviews, impersonation, harassment.
-   **Environmental Impact ("Red AI"):** The high energy consumption of training large models.

---

## 7. Practical Considerations & Future Trends

-   **Model Selection:** Choice of architecture (Encoder, Decoder, Enc-Dec) depends on the task, performance vs. cost trade-offs, and available resources.
-   **Deployment:** Key challenges include model size, latency, throughput, and cost. Optimization techniques (compression, PEFT, KV caching, compilers) are crucial.
-   **Monitoring and Maintenance:** After deployment, it is essential to monitor for performance degradation (drift), track key metrics, and have feedback loops for re-training.
-   **Future Directions:** Sustainable scaling, reducing hallucinations, multimodality (text + image/audio), improved alignment (making models helpful, honest, and harmless), agentic AI (planning, tool use), and trustworthy AI.

---

## 8. Conclusion & Key Takeaways
-   Transformers revolutionized NLP, but pre-training and deployment come with challenges (bias, cost).
-   **Transfer learning (fine-tuning, PEFT)** is the standard for efficiently adapting PTMs.
-   **Efficiency techniques (quantization, distillation, pruning)** are vital for practical deployment.
-   **Rigorous evaluation** requires task-specific metrics, assessment of bias, and human judgment.
-   **XAI and ethical vigilance** are crucial for building trustworthy systems.
-   The field is rapidly evolving towards multimodality, efficiency, better reasoning, and responsible AI.