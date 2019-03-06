from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, CountVectorizer
from pyspark.sql import SparkSession, types
from pyspark.sql.functions import udf, col
from pyspark.ml.clustering import LDA

import sys

spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

comments_schema = types.StructType([
    types.StructField('id', types.StringType(), True),
    types.StructField('comments', types.StringType(), True)
])

def to_text(topic_dist):
    return ','.join(['{:6.5f}'.format(x) for x in topic_dist])

def main():

    processed_path = sys.argv[1]
    output_file = sys.argv[2]

    comm = spark.read.csv(processed_path, schema=comments_schema).repartition(2000).cache()

    tokenizer = Tokenizer(inputCol="comments", outputCol="words")
    # wordsDF = tokenizer.transform(comm)

    hashing = HashingTF(inputCol="words", outputCol="features")
    # count_vect = CountVectorizer(inputCol="words", outputCol="features")
    # cv_model = count_vect.fit(wordsDF)
    # df_features = cv_model.transform(wordsDF)

    # corpus = df_features.select(col('id'), col('features')).cache()

    lda = LDA(k=10, maxIter=10, optimizer='online')

    # lda_model = lda.fit(corpus)
    pipeline = Pipeline(stages=[tokenizer, hashing, lda])
    model = pipeline.fit(comm)

    transformed = model.transform(comm).selectExpr('id', 'topicDistribution')

    topic_text = udf(to_text)
    topics_df = transformed.select(transformed['id'], topic_text(transformed['topicDistribution'])
                                   .alias('topicDistribution'))
    # topics_df.show(truncate=False)

    topics_df.write.option('sep', ',').save(output_file, format='csv', mode='overwrite')


if __name__ == "__main__":
    main()
