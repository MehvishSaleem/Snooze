from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, CountVectorizer, RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession, types, functions as F
from pyspark.ml.clustering import LDA


import sys

spark = SparkSession.builder.appName('Snooze').getOrCreate()

#input_file = '/Users/Mehvish/Documents/SFU/BigDataLab/Metabot/rand3'#sys.argv[1]
input_file = sys.argv[1]
output = sys.argv[2]


def to_text(topic_dist):
    return ','.join([str(x) for x in topic_dist])


comments_schema = types.StructType([
    types.StructField('id', types.StringType(), True),
    types.StructField('comments', types.StringType(), True),])


def main():
    subreddit_group = spark.read.parquet(input_file).repartition(2000)
    # subreddit_group.show()

    #hashing = HashingTF(inputCol="comments", outputCol="features")
    count_vectorizer = CountVectorizer(inputCol="comments", outputCol="features")

    lda = LDA(k=10, maxIter=10, optimizer='online')

    pipeline = Pipeline(stages=[count_vectorizer, lda])
    model = pipeline.fit(subreddit_group)

    predictions = model.transform(subreddit_group).selectExpr('id', 'topicDistribution')

    change_to_str = F.udf(to_text)

    topics_df = predictions.select(predictions['id'], change_to_str(predictions['topicDistribution'])
                                   .alias('topicDistribution'))

    #topics_df.show(20, False)
    topics_df.write.option('sep', ',').save(output, format='csv', mode='overwrite')
# =======
#     topics_df.repartition(2000).write.option('sep', ',').save(output, format='csv', mode='overwrite')
# >>>>>>> Stashed changes


if __name__ == "__main__":
    main()