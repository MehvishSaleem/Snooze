from pyspark.ml.feature import HashingTF, CountVectorizer
from pyspark.sql import SparkSession, types, functions as F
from pyspark.ml.clustering import LDA
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import nltk
import sys


spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

input_comments = 'RC_2016-01-aaaa.json.gz'
output_file = 'fin_out'

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

lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def clean_data(document):
    nltk.data.path.append('/home/namitas/nltk_data')
    doc = []
    for word in document[0]:
        word = word.lower()
        tokens = tokenizer.tokenize(word)
        stemmed_tokens = [lemm.lemmatize(w) for w in tokens if
                          w not in stopwords and w.isalpha()]
        if stemmed_tokens:
            doc.append(','.join(stemmed_tokens))
    return doc


def to_text(topic_dist):
    return ','.join([str(x) for x in topic_dist])


def main():
    comments = spark.read.json(input_comments, schema=comments_schema).repartition(100)
    comm = comments.select(comments['subreddit'].alias('id'), comments['body']).limit(50)
    preprocess = F.udf(clean_data, returnType=types.ArrayType(types.StringType()))
    comm_split = comm.select(comm['id'], F.split(comm['body'], ' ').alias('comments'))
    sub_group = comm_split.groupBy(comm_split['id']).agg(F.collect_list('comments').alias('comments')) \
                                                    .select(F.col('id'), F.col('comments'))

    comm_lemm = sub_group.select(sub_group['id'], preprocess(sub_group['comments']).alias('comments')).cache()

    # hashing_model = HashingTF(inputCol="comments", outputCol="features")
    # result = hashing_model.transform(comm_lemm)

    cv = CountVectorizer(inputCol="comments", outputCol="features")
    count_vectorizer_model = cv.fit(comm_lemm)
    result = count_vectorizer_model.transform(comm_lemm)
    result.show(truncate=False)
    #
    # vocabArray = count_vectorizer_model.vocabulary
    # print(vocabArray)

    corpus = result.select(result['id'], result['features']).cache()

    lda = LDA(k=5, optimizer='online')
    lda_model = lda.fit(corpus)

    transformed = lda_model.transform(corpus)
    transformed.show(truncate=False)

    topic_text = F.udf(to_text)
    topics_df = transformed.select(transformed['id'], topic_text(transformed['topicDistribution'])
                                   .alias('topicDistribution'))
    #topics_df.show(truncate=False)

    topics_df.write.option('sep', ',').save(output_file, format='csv', mode='overwrite')


if __name__ == "__main__":
    main()
