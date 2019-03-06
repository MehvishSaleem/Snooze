from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, CountVectorizer
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.clustering import LDA

import sys

spark = SparkSession.builder.appName('Snooze').getOrCreate()
input_file = sys.argv[1]
output = sys.argv[2]
output_top_topic = sys.argv[3]


def to_text(topic_dist):
    return ','.join([str(x) for x in topic_dist])


def max_topic_prob(topics):
    return max(topics.split(','))


 # specify vector id of words to actual words
def topic_render(terms,vocabList):
    result = []
    for i in range(10):
        term = vocabList[terms[i]]
        result.append(term)
    return result


def main():
    subreddit_group = spark.read.parquet(input_file)

    # hashing = HashingTF(inputCol="comments", outputCol="features")
    count_vectorizer = CountVectorizer(inputCol="comments",
                                       outputCol="features")

    lda = LDA(k=10, maxIter=10, optimizer='online')

    pipeline = Pipeline(stages=[count_vectorizer, lda])
    model = pipeline.fit(subreddit_group)

    topic_dist = model.transform(subreddit_group)\
        .selectExpr('id', 'topicDistribution')

    change_to_str = F.udf(to_text)
    max_topic = F.udf(max_topic_prob)

    topics_df = topic_dist.select(topic_dist['id'],
                                  change_to_str(topic_dist['topicDistribution'])
                                  .alias('topicDistribution'))

    topics_max_prob = topics_df.withColumn('MaxTopicDist',
                                           max_topic('topicDistribution'))

    topics = topics_max_prob.filter(F.col('MaxTopicDist') > '0.7')\
        .select('id', 'topicDistribution')

    topics_indices = model.stages[1].describeTopics(10)
    vocabList = model.stages[0].vocabulary

    get_vocab = F.udf(lambda t: topic_render(t, vocabList))

    top_topics = topics_indices.select(topics_indices['topic'],
                                       change_to_str(get_vocab(topics_indices[
                                    'termIndices'])).alias('term_words'),
                                       change_to_str(
                                           topics_indices['termWeights']).alias(
                                           'term_weight'))

    topics.write.option('sep', ',').save(output, format='csv', mode='overwrite')
    top_topics.write.option('sep', ',').save(output_top_topic, format='csv',
                                             mode='overwrite')


if __name__ == "__main__":
    main()
