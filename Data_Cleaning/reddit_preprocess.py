from pyspark.sql import SparkSession, types, functions as F
from nltk.tokenize import RegexpTokenizer
import nltk
import sys

spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

input_comments = sys.argv[1]
output_unsorted = sys.argv[2]
output_comments = sys.argv[3]
output_ups = sys.argv[4]
output_ratio = sys.argv[5]

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
])

stopwords = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def merge_lists(words):
    return [w for word in words for w in word]


def is_ascii(word):
    return all(ord(char) < 128 for char in word)


def get_ratio(comments, ups):
    return round(comments / ups, 3) if ups != 0 else 0


def clean_data(document):
    nltk.data.path.append('/home/mehvishs/nltk_data')
    doc = []
    for word in document.split():
        word = word.lower()
        tokens = tokenizer.tokenize(word)
        stemmed_tokens = [w for w in tokens if w not in stopwords
                          and w.isalpha() and len(w) > 2 and is_ascii(w)]
        if stemmed_tokens:
            doc.append(','.join(stemmed_tokens))
    return doc


def main():
    change_to_str = F.udf(merge_lists,
                          returnType=types.ArrayType(types.StringType()))

    preprocess = F.udf(clean_data,
                       returnType=types.ArrayType(types.StringType()))

    comment_ups_ratio = F.udf(get_ratio)

    sub_comments = spark.read.json(input_comments,
                                   schema=comments_schema).repartition(500)

    comm = sub_comments.select(sub_comments['subreddit'].alias('id'),
                               sub_comments['body'].alias('comments'),
                               sub_comments['ups'].alias('ups'))

    comm_cleaned = comm.select(comm['id'],
                               preprocess(comm['comments']).alias('comments'),
                               comm['ups'])

    subreddit_group = comm_cleaned.groupBy(comm_cleaned['id']) \
        .agg(change_to_str(F.collect_list('comments')).alias('comments'),
             F.sum('ups').alias('ups')
             , F.count('id').alias('comm_count')).select('id', 'comments',
                                                         'ups',
                                                         'comm_count').cache()

    comments_ups_ratio = subreddit_group.withColumn("comm_ups_ratio",
                                                    comment_ups_ratio(
                                                        'comm_count', 'ups'))

    ups_sorted = subreddit_group.orderBy(subreddit_group.ups.desc()).select(
        'id', 'ups')

    comments_count_sorted = subreddit_group.orderBy(
        subreddit_group.comm_count.desc()).select('id', 'comm_count')

    ratio_sorted = comments_ups_ratio.orderBy(
        comments_ups_ratio.comm_ups_ratio.desc()).select('id', 'comm_ups_ratio')

    subreddit_group.write.format('parquet').save(output_unsorted,
                                                 mode='overwrite')
    ups_sorted.write.format('csv').save(output_ups)
    comments_count_sorted.write.format('csv').save(output_comments)
    ratio_sorted.write.format('csv').save(output_ratio)


if __name__ == "__main__":
    main()
