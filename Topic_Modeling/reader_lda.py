from pyspark.ml.feature import CountVectorizer
from pyspark.sql import types
from pyspark.sql import SparkSession, types, functions as F
from pyspark.mllib.linalg import Vectors, Vector, SparseVector
from pyspark.mllib.clustering import LDA, LDAModel
from nltk.tokenize import RegexpTokenizer
import numpy as np
from pyspark.ml.feature import StringIndexer

from nltk.stem import WordNetLemmatizer

import nltk
import sys

spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

input_comments = sys.argv[1]

input_submission = sys.argv[2]

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType(), True),
    types.StructField('author', types.StringType(), True),
    types.StructField('author_flair_css_class', types.StringType(), True),
    types.StructField('author_flair_text', types.StringType(), True),
    types.StructField('body', types.StringType(), True),
    types.StructField('controversiality', types.LongType(), True),
    types.StructField('created_utc', types.StringType(), True),
    types.StructField('distinguished', types.StringType(), True),
    types.StructField('downs', types.LongType(), True),
    types.StructField('edited', types.StringType(), True),
    types.StructField('gilded', types.LongType(), True),
    types.StructField('id', types.StringType(), True),
    types.StructField('link_id', types.StringType(), True),
    types.StructField('name', types.StringType(), True),
    types.StructField('parent_id', types.StringType(), True),
    types.StructField('retrieved_on', types.LongType(), True),
    types.StructField('score', types.LongType(), True),
    types.StructField('score_hidden', types.BooleanType(), True),
    types.StructField('subreddit', types.StringType(), True),
    types.StructField('subreddit_id', types.StringType(), True),
    types.StructField('ups', types.LongType(), True),
    # types.StructField('year', types.IntegerType(), False),
    # types.StructField('month', types.IntegerType(), False),
])

submissions_schema = types.StructType([
    types.StructField('archived', types.BooleanType(), True),
    types.StructField('author', types.StringType(), True),
    types.StructField('author_flair_css_class', types.StringType(), True),
    types.StructField('author_flair_text', types.StringType(), True),
    types.StructField('created', types.LongType(), True),
    types.StructField('created_utc', types.StringType(), True),
    types.StructField('distinguished', types.StringType(), True),
    types.StructField('domain', types.StringType(), True),
    types.StructField('downs', types.LongType(), True),
    types.StructField('edited', types.BooleanType(), True),
    types.StructField('from', types.StringType(), True),
    types.StructField('from_id', types.StringType(), True),
    types.StructField('from_kind', types.StringType(), True),
    types.StructField('gilded', types.LongType(), True),
    types.StructField('hide_score', types.BooleanType(), True),
    types.StructField('id', types.StringType(), True),
    types.StructField('is_self', types.BooleanType(), True),
    types.StructField('link_flair_css_class', types.StringType(), True),
    types.StructField('link_flair_text', types.StringType(), True),
    types.StructField('media', types.StringType(), True),
    types.StructField('name', types.StringType(), True),
    types.StructField('num_comments', types.LongType(), True),
    types.StructField('over_18', types.BooleanType(), True),
    types.StructField('permalink', types.StringType(), True),
    types.StructField('quarantine', types.BooleanType(), True),
    types.StructField('retrieved_on', types.LongType(), True),
    types.StructField('saved', types.BooleanType(), True),
    types.StructField('score', types.LongType(), True),
    types.StructField('secure_media', types.StringType(), True),
    types.StructField('selftext', types.StringType(), True),
    types.StructField('stickied', types.BooleanType(), True),
    types.StructField('subreddit', types.StringType(), True),
    types.StructField('subreddit_id', types.StringType(), True),
    types.StructField('thumbnail', types.StringType(), True),
    types.StructField('title', types.StringType(), True),
    types.StructField('ups', types.LongType(), True),
    types.StructField('url', types.StringType(), True),
    # types.StructField('year', types.IntegerType(), False),
    # types.StructField('month', types.IntegerType(), False),
])

lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def clean_data(document):
    doc = []
    for word in document:
        word = word.lower()
        tokens = tokenizer.tokenize(word)
        stemmed_tokens = [lemm.lemmatize(w) for w in tokens if
                          w not in stopwords]
        if stemmed_tokens:
            doc.append(','.join(stemmed_tokens))
    return doc


comments = spark.read.json(input_comments, schema=comments_schema)

comm = comments.select(comments['subreddit'].alias('id'),
                       comments['body']).limit(50)

# TODO: Ask about concatenate strings in groupby instead of collect list ?
subreddit_group = comm.groupBy(comm['id']).agg(F.collect_list('body')) \
    .select(F.col('id').alias('id'),
            F.concat_ws(', ', 'collect_list(body)').alias('body'))

clean_data = F.udf(clean_data, returnType=types.ArrayType(types.StringType()))

comm_split = subreddit_group.select(subreddit_group['id'],
                                    F.split(subreddit_group['body'], ' ').alias(
                                        'comments'))
comm_lemm = comm_split.select(comm_split['id'],
                              clean_data(comm_split['comments']).alias(
                                  'comments'))

# comm_lemm.show(truncate=False)

# Indexed Subreddit name so that we can train LDA
indexer = StringIndexer(inputCol="id", outputCol="index")
indexed = indexer.fit(comm_lemm).transform(comm_lemm)

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="comments", outputCol="vectors")
count_vectorizer_model = cv.fit(indexed)
result = count_vectorizer_model.transform(indexed)
# result.show(truncate=False)

corpus = result.select(result['index'].cast('long'), result['vectors']) \
    .rdd.map(lambda x: [x[0], Vectors.fromML(x[1])]).cache()

# # for x in corpus.collect():
# #     print(x)
#
ldaModel = LDA.train(corpus, k=10)
topics = ldaModel.topicsMatrix()

# vocabArray = count_vectorizer_model.vocabulary
print(topics)

# for topic in range(10):
#     print("Topic " + str(topic) + ":")
#     for word in range(0, ldaModel.vocabSize()):
#         print(" " + str(topics[word][topic]))
