from pyspark.sql import SparkSession, types, functions as F
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.feature import Word2Vec




spark = SparkSession.builder.appName('Snooze').getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

input_comments = sys.argv[1]
output = sys.argv[2]

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
    return [j for i in topic_dist for j in i]


lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def clean_data(document):
    nltk.data.path.append('/home/namitas/nltk_data')
    doc = []
    for word in document.split():#[0]:
        word = word.lower()
        tokens = tokenizer.tokenize(word)
        stemmed_tokens = [lemm.lemmatize(w) for w in tokens if
                          w not in stopwords and w.isalpha()]
        if stemmed_tokens:
            doc.append(','.join(stemmed_tokens))
    return doc


def main():
    #input_comments = '/Users/Mehvish/Documents/SFU/BigDataLab/Metabot/comments/RC_2016-01-aaaa.json.gz'
    change_to_str = F.udf(to_text, returnType=types.ArrayType(types.StringType()))

    sub_comments = spark.read.json(input_comments, schema=comments_schema).repartition(500)
    comm = sub_comments.select(sub_comments['subreddit'].alias('id'), sub_comments['body'].alias('comments'),
                               sub_comments['ups'].alias('ups')) #.where(sub_comments['subreddit'] == 'AskReddit').limit(10)

    preprocess = F.udf(clean_data, returnType=types.ArrayType(types.StringType()))

    comm_cleaned = comm.select(comm['id'], preprocess(comm['comments']).alias('comments'), comm['ups'])
    #comm_cleaned.show(truncate=False)

    subreddit_group = comm_cleaned.groupBy(comm_cleaned['id']).agg(change_to_str(F.collect_list('comments')).alias('comments')
                                                                                 , F.sum('ups').alias('ups'),
                                                                                F.count('id').alias('count')) \
                                                    .select('id', 'comments', 'ups', 'count')

    #subreddit_group.show(20, False)
    #print("done")
    subreddit_group.write.format('parquet').save(output, mode='overwrite')


if __name__ == "__main__":
    main()
