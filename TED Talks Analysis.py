import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.text import Text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder

nltk.data.path.append('/Users/zlatabatmanova/nltk_data')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng', download_dir='/Users/zlatabatmanova/nltk_data')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
stop_words = set(stopwords.words('english'))

df_main = pd.read_csv('ted_main.csv')       # Creating a DataFrame for the 1st .csv file
print(df_main.head())
df_trans = pd.read_csv('transcripts.csv')   # Creating DataFrame for the 2nd .csv file containing transcripts
df_trans.describe()

'''Extracting topics from the tags column for a WordCloud'''
def parse_tags(tags_value):
    '''Extracts individual tags from different columns and from different formats'''
    if isinstance(tags_value, list):                                                        # Use isinstance() function to check only the tags saved as lists ('tags_value', to be precise)
        return [tag.strip() for tag in tags_value if isinstance(tag, str) and tag.strip()]  # Return a list of tags as strings without any leading or trailing whitespaces; ignore all non-strings
    elif isinstance(tags_value, str) and tags_value.strip():
        return re.findall(r"[\"']([^\"']+)[\"']", tags_value)                               # Use regEx to find the tags within quotation marks and extract them
    else:
        return None


def tags_frequency(df_main):
    '''Calculates the frequency of each tag. Returns a list of tags with a number of occurrences'''
    all_tags = []                                           # Create an empty list
    for tags_value in df_main['tags']:                      # Loop over each row in 'tags' column and return tags
        parsed_tags = parse_tags(tags_value)                # Assign the individual tags from the parse_tags() function to a value
        all_tags.extend(parsed_tags)                        # Add the tags to the all_tags list
    print(f"Overall number of tags found: {len(all_tags)}") # returns a list of tags, including duplicates
    return Counter(all_tags)                                # a Counter (a dict subclass for counting hashable objects) returns the frequency of each unique tag


def tag_freq(tag_frequencies):
    '''Counts and saves frequencies'''
    if tag_frequencies:
        print(f"A number of unique tags: {len(tag_frequencies)}")
        print(f"Top 10 topics:")
        sorted_tags = sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True) # Create a variable with a list of tag-frequency pairs, sorted from most to least frequent (with the help of lamda function)
        top_10_df = pd.DataFrame(sorted_tags[:10], columns=['tag', 'frequency'])        # Create a new DataFrame with 2 columns called 'tag' and 'frequency' for the first 10 topics sorted out in the previous line
        for tag, count in sorted_tags[:10]:                                             # Print the output for the top 10 tag-frequency pairs
            print(f"{tag}: {count}")
        return top_10_df
    else:
        print("No tags found")
        return pd.DataFrame()

def create_wordcloud_topics(tag_frequencies):
    '''Creates a WordCloud for the topics range'''
    textx = " ".join(tag_frequencies)        # Create a string of tag names
    wordcloud=WordCloud(background_color='white', max_words=100).generate(textx)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def create_topics_barplot(tag_frequencies):
    ''' Creates a barplot for the most popular topics'''
    plt.figure(figsize=(10,6))
    sns.barplot(x='frequency', y='tag', hue='tag', data=df_top_tags, palette="rocket", legend=False)
    plt.title('Top 10 TED Talk Topics')
    plt.xlabel('Frequency')
    plt.ylabel('Topic')
    plt.show()

def time_conversion(df_main):
    ''' Converts Unix data format to a standard one'''
    unix_to_normal = pd.to_datetime(df_main['film_date'], unit='s') # Convert Unix timestamps in seconds
    film_year = unix_to_normal.dt.year                              # Extract the years

    df_main['year'] = film_year              # Assign a new column 'film_year' to the DataFrame
    pd.set_option('display.max_rows', 100)   # Show the first 100 rows in the DataFrame
    df_main.sort_values(by='year')           # Sort the df in chronological order

    # Count an average video duration per each year
    df_main['duration'] = df_main['duration'] / 60                                             # Convert seconds to minutes
    df_main['duration'] = df_main['duration'].round(2)                                         # Round the result to 2 decimal points
    avr_duration = df_main.groupby('year')['duration'].mean().reset_index(name='avr_duration') # Calculate an average video duration per year and return it back to the df column 'avr_duration'
    df_avr_duration = pd.DataFrame(avr_duration)                                               # Create a df and assign it to a variable 'df_avr_duration'
    return df_avr_duration

def duration_trends(df_avr_duration):
    '''Creates a linear diagram showing the duration trends over the years'''
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_avr_duration, x='year', y='avr_duration', marker='o', color='darkred', linewidth=2)
    plt.title("TED Talks Duration Trends Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Length (minutes)")
    plt.grid(True, color='black')
    plt.show()

'''NLTK part'''
def transcripts_cleaning(s):
    '''Cleans transcripts in df_trans'''
    cleaned_s = re.sub(r"[^\w\s]", '', s)                                              # Remove everything from the string that is not a letter, digit, underscore, or whitespace
    cleaned_s = re.sub(r"\s+", ' ', cleaned_s)                                         # Replace multiple whitespace characters with a single space
    tok_transcript = word_tokenize(cleaned_s)                                          # Tokenize the cleaned text
    filtered_tok = [word for word in tok_transcript if word.lower() not in stop_words] # Remove all stopwords from the text
    return (' '.join(filtered_tok))                                                    # Create a string of filtered tokens

def create_cs_column(df_trans):
    '''Creates a new column with cleaned transcripts within df_trans'''
    df_trans['cleaned_speeches'] = df_trans['transcript'].apply(transcripts_cleaning)

# Extract the most frequently used words by the speaker with most performances
def performer_words(df_main, df_trans):
    '''Extracts a person who performed the most talks and clean their speeches'''
    who = df_main['main_speaker'].value_counts()                                    # Extract a speaker with the most frequent performances -> The top performer is Hans Rosling with 9 talks
    hans_rosling_url = df_main[df_main['main_speaker'] == 'Hans Rosling']['url']    # Find the URLs of his talks in df_main to match them with the URLs and transcripts in df_trans
    hans_rosling_talks = df_trans[df_trans['url'].isin(hans_rosling_url)]           # Match his speeches in transcripts.csv
    hr = hans_rosling_talks['transcript'].apply(transcripts_cleaning)               # Clean his speeches, applying the 'transcripts_cleaning' function
    return who, hr

def concord_global(hr):
    ''' Creates a concordance of the word "global"'''
    hr_all = ' '.join(hr)                                               # Join the cleaned transcripts to a string
    tokens = word_tokenize(hr_all)                                      # Tokenize the scripts
    hr_transcripts = Text(tokens)                                       # Create an object for NLTK
    return hr_all, hr_transcripts, hr_transcripts.concordance('global') # Generate a concordance for the word 'global

def hans_rosling_top_words(hr_transcripts):
    '''Creates a list of the most frequently used words'''
    hr_vocab = hr_transcripts.vocab()          # Return the frequency distribution
    hr_words = list(hr_vocab.keys())           # Return a list of the words without their frequencies
    return hr_vocab

def rosling_plot_analysis(hr_transcripts):
    ''' Creates a dispersion and a frequency distribution plots for Hans Rosling's talks'''
    # Dispersion Plot
    hr_transcripts.dispersion_plot(["countries", "child", "income", "population", "world"])

    # Frequency Distribution Plot
    hr_all = ' '.join(hr_transcripts.tokens)     # Collect all tokens in one string
    tokens = word_tokenize(hr_all)               # Tokenize it all
    freq_dist = FreqDist(tokens)                 # Create a frequency distribution for the given tokens
    plt.figure(figsize=(12, 7))                  # Create a frequency distribution plot
    freq_dist.plot(30, title="Top 30 Most Frequent Words in Hans Rosling's Talks")
    plt.show()

# COLLOCATIONS FROM THE FUNNIEST TALKS PART
def unpack_ratings(df_main):
    ''' Unpacks the ratings data'''
    df_main['ratings'] = df_main['ratings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df_main

# Extract the ratings from the nested ratings column and create separate columns for each category
def rating_sum(d):
    '''Counts the "Count"-values within the ratings, e.g., [{'Name': 'Courageous', 'Count': 760}, ...] '''
    if isinstance(d, list):        # Take a list of dictionaries, containing 'Count'
        counts = 0                 # Create a variable to hold the sum of all 'count' values
        for obj in d:              # Loop through each dictionary in the list
            counts += obj['count'] # Add value connected to 'Count' key to the 'counts' variable
        return counts              # Return a sum of all 'Count' values

def review_cat(df_main):
    ''' Creates new columns for each review category and for the maximum score'''
    for i, rate in enumerate(df_main['ratings']):          # Loop over each element in the 'ratings' column, assigning an index to it
        if isinstance(rate, list):                         # Continue only if the 'rate' value is a list
            maximum = ["", 0]                              # Initialize a list to store the name and count of the highest-rated category: [category_name, count]
            for obj in rate:                               # Loop over each dictionary in the ratings list
                name = obj['name'].lower()                 # Convert the rating category name to lowercase for use as a column name
                df_main.loc[i, name] = obj['count']        # Set the value in the corresponding rating column for this row to the 'count' value
                if obj['count'] > maximum[1]:
                    maximum = [name, obj['count']]         # Update the maximum if the count is greater than the highest
            df_main.loc[i, 'top_rated'] = maximum[0]       # After checking all categories, assign the name of the top-rated category to the 'top_rated' column for this row
    return df_main

def count_the_rating(df_main):
    '''Counts overall rating'''
    df_main["overall_rating_count"] = df_main["ratings"].apply(rating_sum)
    return df_main

def create_more_dfs(df_main):
    '''Creates a new DataFrame with the rating names as columns'''
    rating_cols = ['funny', 'ingenious', 'jaw-dropping', 'inspiring', 'beautiful', 'courageous', 'fascinating', 'persuasive']
    rating_df = df_main[rating_cols]
    return rating_df

    # Create a df with the talks titles, separate 'funny' reviews column, and its transcripts
    #Merging 'funny' reviews, transcripts and its titles in one df
def fun_together(df_main, df_trans):
    '''Merging 'funny' reviews, transcripts and its titles in one df (using URLs), cleans the scripts,
    extracts the top 10 funniest talks and returns their cleaned transcripts'''
    funny_talks_df = df_main[df_main['funny'] > 100]    # Filter the talks with more than 100 votes
    funny_talks_url = funny_talks_df['url']
    funny_talks = df_trans[df_trans['url'].isin(funny_talks_url)]
    merged_funny = funny_talks.merge(funny_talks_df, on='url')

    sorted_fun = merged_funny.sort_values(by='funny', ascending=False).head(10)  # Top 10 selection
    top_fun_transcripts = sorted_fun['transcript'].tolist()

    cleaned_top_fun = [transcripts_cleaning(t) for t in top_fun_transcripts]     # Cleaning the selected talks
    return cleaned_top_fun

def fun_collocations_collect(cleaned_top_fun):
    '''Collects the collocations with NLTK for all 10 talks '''
    funny_texts = ' '.join(cleaned_top_fun)    # Create a string of top 10 cleaned transcripts (the funny ones)
    tokens = word_tokenize(funny_texts)        # Tokenize them
    top = nltk.text.Text(tokens)
    return funny_texts, top

def collocations_heatmap(funny_texts):
    ''' Identifies collocations (bigrams), selects the top 20 most frequent bigrams based on PMI (Pointwise Mutual Information), and
    creates a Seaborn co-occurrences matrix, showing how often the analyzed word pairs appear together'''
    #Collocations with Bigram measures
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    tokens2 = word_tokenize(funny_texts.lower())                              # Tokenize the scripts (already a sting)
    bcf = BigramCollocationFinder.from_words(tokens2)                         # Identify bigrams
    collocations = bcf.nbest(nltk.collocations.BigramAssocMeasures().pmi, 20) # Select the top 20 most frequent bigrams, PMI related collocations

    # Counting bigrams for a heatmap
    bigrams = list(nltk.bigrams(tokens2))
    bigram_freq = Counter(bigrams)
    top_bigram = bigram_freq.most_common(20)

    # Create a DataFrame for a heatmap (in a long data format to convert it into a matrix one after that)
    rows = []
    for ((w1, w2), freq) in top_bigram:
        rows.append((w1, w2, freq))
    df_bigram = pd.DataFrame(rows, columns=['word1', 'word2', 'freq'])

    # Pivot for the heatmap (reshapes the df to a matrix format)
    bigram_matrix = df_bigram.pivot(index='word1', columns='word2', values='freq').fillna(0)

    # Visualize the results with a Seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(bigram_matrix, annot=True, fmt='.0f', cmap='icefire')
    plt.title("20 Most Often Occurring Collocations Heatmap")
    plt.ylabel("First Word")
    plt.xlabel("Second Word")
    plt.tight_layout()
    plt.show()

def funcom_corr(df_main):
    ''' Counts correlation between 'funny' evaluation and a number of comments and visualizes it'''
    df_filtered = df_main.dropna(subset=['comments', 'funny']) # Create a df and remove all rows that have NaNs in either the 'comments' or 'funny' columns
    comments = df_filtered['comments'].to_numpy()              # Convert the resulting series into a NumPy array
    funny = df_filtered['funny'].to_numpy()
    correlation = np.corrcoef(comments, funny)[0, 1]           # Return the Pearson correlation coefficient between the number of comments and the 'funny' rating

    # Correlation visualization  with lmplot from the Seaborn library
    sns.lmplot(data=df_main, x='funny', y='comments',  aspect=2, line_kws={'color': 'darkred'}, scatter_kws={'color': 'darkblue', 'alpha':0.6})
    plt.title('Correlation Between a Number of Comments and Evaluation of a Talk as "Funny"')
    plt.xlabel('Evaluated as "funny"')
    plt.ylabel('Number of comments')
    plt.tight_layout()
    plt.show()
    return print(f"Correlation between funny evaluation and number of comments: {correlation:.2f}")
    
    # NER in the most commented Talks
def viewed_talks(df_main):
    # Extract top 10 most viewed Talks # Concatenate the top 10 talks with maximum views with the scripts
    top10_views = df_main.sort_values(by='views', ascending=False).head(10)

    top_views = top10_views.merge(df_trans[['url', 'transcript']], on='url', how='left')
    top_views['cleaned_for_views'] = top_views['transcript'].apply(transcripts_cleaning)
    k = top_views['cleaned_for_views']
    return top_views, top10_views, k

def process(k):
    '''Takes the cleaned transcripts, joins them to a string, tokenizes them, POS-tags them and applies NER'''
    tv = " ".join(k)
    sent = nltk.word_tokenize(tv)
    sent = nltk.pos_tag(sent)  # POS-tag the tokenized transcripts
    tree = nltk.ne_chunk(sent) # Apply NER for the tokenized transcripts
    return sent, tree

def named_entities_extraction(tree, label):
    '''Extracts named entities of a particular label and returns them'''
    named_entities = []
    for subtree in tree.subtrees():                # Go through all subtrees in the chunk tree
        if subtree.label() == label:               # If the subtree is labeled with the desired entity type
            entity_words = []                      # Create a list to store the entity words
            for leaf in subtree.leaves():          # Get each word in the entity
                entity_words.append(leaf[0])       # leaf[0] is the actual word, whereas leaf[1] is a POS tag
            named_entity = " ".join(entity_words)  # Join the words into a single string
            named_entities.append(named_entity)    # Add it to the initial list
    return named_entities

# Apply it all:
if __name__ == "__main__":
    tag_frequencies = tags_frequency(df_main)
    df_top_tags = tag_freq(tag_frequencies)
    print("\nDataFrame:")
    print(df_top_tags)                           # Print a list of the most popular TED Talks tags
    create_wordcloud_topics(tag_frequencies)     # Generate a WordCloud with the Talks topics
    create_topics_barplot(tag_frequencies)       # Generate a bar plot for the topics trends overview
    time_conversion(df_main)                     # Convert Unix time to a standard one
    df_avr_duration = time_conversion(df_main)
    duration_trends(df_avr_duration)             # Generate a linear plot with Talks duration trends over of the decades
    create_cs_column(df_trans)
    who, hr = performer_words(df_main, df_trans) # Unpack the tuple to use the values separately
    print(who)                                   # Print the speakers with a number of the talks given
    hr_all, hr_transcripts, global_concordance = concord_global(hr)
    hr_vocab = hans_rosling_top_words(hr_transcripts)
    print("\nThe most common words in these talks are the following:", hr_vocab.most_common(50)) # Generate frequency distribution for the top 50 words used by Hans Rosling's
    print(global_concordance)                      # Print the concordance of the word 'global'
    hans_rosling_top_words(hr_transcripts)
    rosling_plot_analysis(hr_transcripts)          # Generate a dispersion plot and a frequency distribution plot for the most often used words by Hans Rosling
    unpack_ratings(df_main)
    rating_sum(df_main)
    review_cat(df_main)
    count_the_rating(df_main)
    create_more_dfs(df_main)
    fun_together(df_main, df_trans)
    cleaned_top_fun = fun_together(df_main, df_trans)
    fun_collocations_collect(cleaned_top_fun)
    funny_texts, top = fun_collocations_collect(cleaned_top_fun)
    print("\nUp to twenty collocations")                         # Print the collocations from the talks, which are rated as the funniest ones
    top.collocations()
    print("\nUp to a hundred collocations")
    top.collocations(num=100)
    print("\nCollocations that might have one word in between")
    top.collocations(window_size=3)
    collocations_heatmap(funny_texts)                             # Create a collocations heatmap
    funcom_corr(df_main)                                          # 'funny'-'comments' correlation
    top_views, top10_views, k = viewed_talks(df_main)
    process(k)                                                    # NER output for 'PERSON' und 'ORGANIZATION' labels
    sent, tree = process(k)
    label = 'PERSON'
    person = named_entities_extraction(tree, label)
    print("\nNamed entities of 'PERSON':")
    print(named_entities_extraction(tree, 'PERSON'))
    label = 'ORGANIZATION'
    print("\nNamed entities of 'ORGANIZATION':")
    print(named_entities_extraction(tree, 'ORGANIZATION'))











