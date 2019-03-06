import string

import nltk
from nltk import WordNetLemmatizer
from pyspark.sql import SparkSession, types
from pyspark.sql.functions import col, concat_ws, collect_list, udf
import sys


spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

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
    types.StructField('ups', types.LongType(), True)
])

lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

def clean_data(document):
    nltk.data.path.append('/home/namitas/nltk_data')
    doc = document.strip().lower()
    for c in string.punctuation:
        doc = doc.replace(c, '')
    stemmed_tokens = [lemm.lemmatize(w) for w in doc.split() if
                          w not in stopwords and w.isalpha()]

    processed_doc = ' '.join(stemmed_tokens)

    return processed_doc

def main():
    # input_comments = '/Users/Mehvish/Documents/SFU/BigDataLab/Metabot/comments/RC_2016-01-aaaa.json.gz'
    # input_comments = '../BigData-Snooze/RC_2016-01-aaaa.json.gz'
    # output_file = 'comments_cleaned'

    input_comments = sys.argv[1]
    output_file = sys.argv[2]

    sub_comments = spark.read.json(input_comments, schema=comments_schema).repartition(2000)
    comm = sub_comments.select(sub_comments['subreddit'].alias('id'), sub_comments['body'].alias('comments'))#.limit(10)

    text_processor = udf(clean_data)
    comm_cleaned = comm\
                    .withColumn('comments_cleaned', text_processor(col('comments')))\
                    .select(col('id'), col('comments_cleaned').alias('comments'))

    comm_cleaned = comm_cleaned.filter(col('comments') != '')

    comm = comm_cleaned\
                    .groupBy(col('id'))\
                    .agg(concat_ws(' ', collect_list(col('comments'))).alias('comments'))

    # comm.printSchema()

    comm.write.option('sep', ',').save(output_file, format='csv', mode='overwrite')


if __name__ == "__main__":
    main()