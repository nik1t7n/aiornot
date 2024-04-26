import React from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yLight } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "../components/CodeBlock";

const DocsPage = () => {
  const codeClasses = "rounded-lg shadow-lg bg-gray-100 p-4";
  return (
    <div className="container mx-auto px-4 py-8 mt-24">
      <h1 className="text-3xl font-bold mb-4">How the model works?</h1>
      <p>
        I have preprocessed and trained a model using Jupyter Notebook, so let’s
        figure out how it works!
      </p>

      <h2 className="text-2xl font-bold mt-8 mb-4">Data</h2>
      <p className="drop-shadow-md text-red-600">
        [You MUST be logged in to Kaggle to Acces Dataset]
      </p>
      <br />
      <p>
        Link:{" "}
        <a
          href="https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text"
          target="_blank"
          rel="noopener noreferrer"
          className="drop-shadow-md text-green-600 underline decoration-solid"
        >
          AI Vs Human Text dataset
        </a>{" "}
        <br />
        It has 500K essays both created by AI and written by Human.
      </p>

      <h2 className="text-2xl font-bold mt-8 mb-4">Imports</h2>
      <CodeBlock
        code={`
  import pandas as pd
  import numpy as np
  import seaborn as sns
  import string
  import nltk
  from nltk.corpus import words
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.svm import SVC
  from sklearn.metrics import classification_report
  import joblib
      `}
      />

      <h2 className="text-2xl font-bold mt-8 mb-4">
        Reading and Exploring Data
      </h2>
      <CodeBlock
        code={`
  df = pd.read_csv("AI_Human.csv")
  df.info()
  df.describe()
  sns.countplot(data=df, x='generated')
  print('Total Texts:', df['generated'].count())
  print('Human Written Texts:', (df['generated'] == 0.0).sum())
  print('AI Generated Texts:', (df['generated'] == 1.0).sum())
      `}
      />

      <h2 className="text-2xl font-bold mt-8 mb-4">Data Preprocessing</h2>
      <p>
        After comprehending the dataset, the next step is to preprocess all
        texts within the dataset to make it easier for the model to train.
      </p>

      <h3 className="text-xl font-bold mt-6 mb-2">
        1. Remove Unnecessary Tags
      </h3>
      <CodeBlock
        code={`
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_words= ' '.join(filtered_words)
    return filtered_words

df['text']=df['text'].apply(remove_stopwords)
      `}
      />

      <h3 className="text-xl font-bold mt-6 mb-2">
        2. Remove All Punctuation Signs
      </h3>
      <CodeBlock
        code={`
  def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text
      
  df['text']=df['text'].apply(remove_punc)
      `}
      />

      <h3 className="text-xl font-bold mt-6 mb-2">3. Remove Stop Words</h3>
      <CodeBlock
        code={`
  from nltk.corpus import stopwords
  nltk.download('stopwords')

  def remove_stopwords(text):
      stop_words = set(stopwords.words('english'))
      words = nltk.word_tokenize(text)
      filtered_words = [word for word in words if word.lower() not in stop_words]
      filtered_words= ' '.join(filtered_words)
      return filtered_words

  df['text']=df['text'].apply(remove_stopwords)
      `}
      />

      <h2 className="text-2xl font-bold mt-8 mb-4">Model Creation</h2>
      <p>The next step is to create a model pipeline.</p>

      <h3 className="text-xl font-bold mt-6 mb-2">
        Dividing and Preparing Dataset
      </h3>
      <CodeBlock
        code={`
  y=df['generated']
  X=df['text']  
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
      `}
      />

      <h3 className="text-xl font-bold mt-6 mb-2">Pipeline Configuration</h3>
      <CodeBlock
        code={`
  pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),  # Step 1: CountVectorizer
    ('tfidf_transformer', TfidfTransformer()),  # Step 2: TF-IDF Transformer
    ('naive_bayes', MultinomialNB())])
      `}
      />

      <p className="mt-4">
        Here, we configure a pipeline that automatically preprocesses input data
        and then makes predictions without human intervention.
      </p>

      <div className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mt-8 mb-4">
          Understanding the Pipeline
        </h2>
        <p>
          Stop stop stop, you may ask me:{" "}
          <span className="drop-shadow-md text-red-500">
            “What the hell is going on here?”
          </span>
          . I understand you, I had the same feelings at the beginning. So,
          let’s break this out!
          <br />
          <br />
          Here we are{" "}
          <span className="drop-shadow-md text-green-600">
            configuring a Pipeline that will work for us
          </span>{" "}
          as an automatic factory that preprocess input data and then make a
          prediction. And yes – all this features work without human
          intervention.
          <br />
          <br />
          Now I suggest you to go through all of these 3 parts of the Pipeline
          and see what each of the does:
        </p>

        <h3 className="text-xl font-bold mt-6 mb-2">1. CountVectorizer</h3>
        <p>
          <span className="drop-shadow-md text-red-600">“CountVectorizer”</span>{" "}
          is a scikit-learn library method that transforms text data into the
          token (word) counting matrix. For example, there are two following
          sentences:
          <br />
          <br />
          <span className="drop-shadow-md">1) “Cats like milk”</span>
          <br />
          <span className="drop-shadow-md">
            2) “Dogs like playing with a ball”
          </span>
          <br />
          <br />
          “CountVectorizer”{" "}
          <span className="drop-shadow-md text-green-600">
            will transform each sentence into a vector
          </span>{" "}
          where each vector element represents the frequency of occurrence of a
          word in a sentence. A dictionary with unique words will look like:{" "}
          <span className="drop-shadow-md text-orange-600">
            ["cats", "like", "milk", "dogs", "playing", "with", "a", "ball"]
          </span>
          .
          <br />
          <br />
          The vectorized “Cats like milk” will be transformed into the{" "}
          <span className="drop-shadow-md text-orange-600">
            [1, 1, 1, 0, 0, 0, 0, 0]
          </span>
          . While "Dogs like playing with a ball" into the{" "}
          <span className="drop-shadow-md text-orange-600">
            [0, 1, 0, 1, 1, 1, 1, 1]
          </span>
          .
        </p>

        <h3 className="text-xl font-bold mt-6 mb-2">2. TfidfTransformer</h3>
        <p>
          <span className="drop-shadow-md text-red-600">
            “TfidfTransformer”
          </span>{" "}
          is a method that transforms token counting matrix to a weighted matrix
          TF-IDF (Term Frequency-Inverse Document Frequency).
          <br />
          <br />
          TF-IDF is a statistical measure that{" "}
          <span className="drop-shadow-md text-green-600">
            evaluates the importance of a word in the context of a document
          </span>{" "}
          relative to the entire corpus of documents.
          <br />
          The process includes the following steps:
          <br />
          <br />
          <span className="drop-shadow-md text-orange-600">1)</span> Calculating
          the frequency (the term Frequency) for each word in each sentence.
          T.F. is equal to the number of words waiting in the sentence,
          especially on the total number of words in the sentence.
          <br />
          <br />
          <span className="drop-shadow-md text-orange-600">2)</span> Calculating
          the IDF (Inverse Document Frequency) for each word in the text. ISO is
          equal to the logarithm of the ratio of the total number of documents
          to the number of documents containing the word.
          <br />
          <br />
          <span className="drop-shadow-md text-orange-600">3)</span> Calculating
          TF-IDF for each word in each sentence as a product of TF and IDF.
          <br />
          <br />
          For example, the sentence "Cats love milk" can be converted to a
          TF-IDF vector{" "}
          <span className="drop-shadow-md text-orange-600">
            [0.306, 0.306, 0.408, 0, 0, 0, 0, 0]
          </span>
          , The sentence "Dogs like to play with a ball" - in TF-IDF vector{" "}
          <span className="drop-shadow-md text-orange-600">
            [0, 0.306, 0, 0.408, 0.408, 0.408, 0.408, 0.408]
          </span>
          .
        </p>

        <h3 className="text-xl font-bold mt-6 mb-2">3. MultinomialNB</h3>
        <p>
          <span className="drop-shadow-md text-red-600">“MultinomialNB”</span> -
          The multinomial naive Bayesian classifier is one of the machine
          learning methods that is used to solve classification problems,
          especially for text data. It is based on Bayes' theorem and assumes
          that each feature (word) in the text is independent of other features.
          <br />
          <br />
          You can check perfect explanation of how this model works on the video
          by StatQuest:
          <br />
          <br /> 
          <a
            href="https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer"
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-600 underline decoration-solid"
          >
             Naive Bayes, Clearly Explained!!!
          </a>
           
        </p>
      </div>

      <h2 className="text-2xl font-bold mt-8 mb-4">
        Model Training and Evaluation
      </h2>
      <p>
        The next steps involve fitting, predicting, and evaluating the model.
      </p>

      <h3 className="text-xl font-bold mt-6 mb-2">Fitting</h3>
      <CodeBlock
        code={`
  pipeline.fit(X_train, y_train)
      `}
      />

      <h3 className="text-xl font-bold mt-6 mb-2">Predicting</h3>
      <CodeBlock
        code={`
  y_pred = pipeline.predict(X_test)
      `}
      />

      <h3 className="text-xl font-bold mt-6 mb-2">Evaluating</h3>
      <CodeBlock
        code={`
  print(classification_report(y_test, y_pred))
      `}
      />

      <br />

      <p>Output:</p>

      <div className={`container mx-auto px-4 py-8 overflow-x-auto`}>
        <table className={`min-w-full divide-y divide-gray-200 ${codeClasses}`}>
          <thead>
            <tr>
              <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Metric
              </th>
              <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Precision
              </th>
              <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Recall
              </th>
              <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                F1-score
              </th>
              <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Support
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            <tr>
              <td className="px-6 py-4 whitespace-nowrap">0.0</td>
              <td className="px-6 py-4 whitespace-nowrap">0.93</td>
              <td className="px-6 py-4 whitespace-nowrap">0.99</td>
              <td className="px-6 py-4 whitespace-nowrap">0.96</td>
              <td className="px-6 py-4 whitespace-nowrap">91597</td>
            </tr>
            <tr>
              <td className="px-6 py-4 whitespace-nowrap">1.0</td>
              <td className="px-6 py-4 whitespace-nowrap">0.99</td>
              <td className="px-6 py-4 whitespace-nowrap">0.87</td>
              <td className="px-6 py-4 whitespace-nowrap">0.93</td>
              <td className="px-6 py-4 whitespace-nowrap">54574</td>
            </tr>
            <tr>
              <td className="px-6 py-4 whitespace-nowrap">Accuracy</td>
              <td className="px-6 py-4 whitespace-nowrap"></td>
              <td className="px-6 py-4 whitespace-nowrap"></td>
              <td className="px-6 py-4 whitespace-nowrap">0.95</td>
              <td className="px-6 py-4 whitespace-nowrap">146171</td>
            </tr>
            <tr>
              <td className="px-6 py-4 whitespace-nowrap">Macro avg</td>
              <td className="px-6 py-4 whitespace-nowrap">0.96</td>
              <td className="px-6 py-4 whitespace-nowrap">0.93</td>
              <td className="px-6 py-4 whitespace-nowrap">0.94</td>
              <td className="px-6 py-4 whitespace-nowrap">146171</td>
            </tr>
            <tr>
              <td className="px-6 py-4 whitespace-nowrap">Weighted avg</td>
              <td className="px-6 py-4 whitespace-nowrap">0.95</td>
              <td className="px-6 py-4 whitespace-nowrap">0.95</td>
              <td className="px-6 py-4 whitespace-nowrap">0.95</td>
              <td className="px-6 py-4 whitespace-nowrap">146171</td>
            </tr>
          </tbody>
        </table>
      </div>

      <br />

      <p>
        The classification report provides precision, recall, F1-score, and
        support metrics for each class, as well as macro and weighted averages.
      </p>

      <h2 className="text-2xl font-bold mt-8 mb-4">
        Using the Model Outside the Notebook
      </h2>
      <p>
        If we want to use the model outside the notebook, we need to prepare the
        data before inputting it into the model.
      </p>

      <br />

      <CodeBlock
        code={`
  # Example of preparing custom text for prediction
  custom_text = "In simple words"
  count_vectorizer = pipeline.named_steps['count_vectorizer']
  tfidf_transformer = pipeline.named_steps['tfidf_transformer']
  transformed_text = count_vectorizer.transform([custom_text])
  transformed_text = tfidf_transformer.transform(transformed_text)
  prediction = pipeline.named_steps['naive_bayes'].predict(transformed_text)
  print("Prediction:", prediction)
      `}
      />

      <br />

      <p>Similarly, we can dump and load the model for future use.</p>

      <br />

      <CodeBlock
        code={`
  joblib.dump(pipeline, 'AI_Human_Prediction_Model.pkl')
  pipeline = joblib.load('AI_Human_Prediction_Model.pkl')
    `}
      />

      <br />

      <p>
        That's all that I wanted to cover in this article. Thank you for
        reading!
      </p>

      <a
        href="https://disk.yandex.com/d/79ae-p-RUGe9Qw"
        target="_blank"
        rel="noopener noreferrer"
        className="inline-block bg-lime-600 hover:bg-lime-700 text-white font-medium py-2 px-4 rounded mt-4 shadow-md transition duration-300 hover:bg-amber-600"
      >
        Download Ipynb
      </a>
    </div>
  );
};

export default DocsPage;
