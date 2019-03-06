There are 5 Python programs here :

 - `reddit_preprocess.py` Preprocesses the data using NLTK and groups comments of the subreddit.
 This code generates 4 files : 
 
    1. Subreddits and comments file: it is a parquet file which is used as a 
    input in the `topic_modeling.py` file. 
    2. Up Votes from high to low : it is a csv file which has a number of 
    upvotes sorted high to low.
    3. Number of comments: it is a csv file has number of comments sorted high to low. 
    4. Sorted Ratio : Ratio of upvotes to comments 
    
 - `topic_modeling.py` Implementation of LDA Model. This code generates 2 files :
   
    1. Topic Distribution. 
    2. Top words of each topic.
   
   For visualization, please refer to one_page.html (/webapp/templates/)
   For t-SNE model training and fitting, please refer to TSNE_formulate.py (/webapp/)
