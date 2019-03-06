from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, CountVectorizer, RegexTokenizer, \
    StopWordsRemover, CountVectorizerModel
from pyspark.sql import SparkSession, types, functions as F
from pyspark.ml.clustering import LDA
from pyspark.sql.types import ArrayType, StringType

import sys

spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

input_comments = sys.argv[1]
output_file = sys.argv[2]
output_top_topic = sys.argv[3]

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


def to_text(topic_dist):
    return ','.join([str(x) for x in topic_dist])


def topic_render(terms, vocabList):  # specify vector id of words to actual words
    # terms = topic.termIndices
    result = []
    for i in range(10):
        term = vocabList[terms[i]]
        result.append(term)
    return result


def main():
    change_to_str = F.udf(to_text)

    reddit_data = spark.read.json(input_comments, schema=comments_schema).limit(10)
    subreddit = reddit_data.select(reddit_data['subreddit'].alias('id'), reddit_data['body'].alias('comments'))
    # subreddit.show(truncate=False)

    corpus = subreddit.groupBy(subreddit['id']).agg(change_to_str(F.collect_list('comments')).alias('comments'))
    # corpus.show(truncate=False)

    regexTokenizer = RegexTokenizer(inputCol="comments", outputCol="words", pattern="[\\W_]+", minTokenLength=4)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")

    # hashing = HashingTF(inputCol="filtered", outputCol="features")
    count_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
    lda = LDA(k=10, maxIter=10, optimizer='online')

    pipeline = Pipeline(stages=[regexTokenizer, remover, count_vectorizer, lda])
    model = pipeline.fit(corpus)

    predictions = model.transform(corpus).selectExpr('id', 'topicDistribution')
    # predictions.show(truncate=False)
    topics_df = predictions.select(predictions['id'], change_to_str(predictions['topicDistribution']).alias('topicDistribution'))
    topics_df.show(truncate=False)
    # topics_df.write.option('sep', ',').save(output_file, format='csv',
    #                                         mode='overwrite')

    # model.stages[3] - change the indices here based on stanges in the pipeline
    topics_indices = model.stages[3].describeTopics(10)
    vocabList = model.stages[2].vocabulary

    get_vocab = F.udf(lambda t: topic_render(t, vocabList))

    top_topics = topics_indices.select(topics_indices['topic'],
                                         change_to_str(get_vocab(topics_indices['termIndices'])).alias('term_words'),
                                         change_to_str(topics_indices['termWeights']).alias('term_weight'))
    top_topics.show(truncate=False)
    top_topics.write.option('sep', ',').save(output_top_topic, format='csv', mode='overwrite')


if __name__ == "__main__":
    main()
