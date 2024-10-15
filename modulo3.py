import numpy as np
import pandas as pd

# Load the dataset with TweetPos and TweetNeg from Module 2
data = pd.read_csv('dataset_con_afinn_puntajes.csv')

# Define the triangular membership function
def triangular_membership(x, d, e, f):
    if x < d:
        return 0
    elif d <= x <= e:
        return (x - d) / (e - d)
    elif e < x <= f:
        return (f - x) / (f - e)
    else:
        return 0

# Get the global minimum, maximum, and mid values for TweetPos and TweetNeg
tweet_pos_min = data['TweetPos'].min()
tweet_pos_max = data['TweetPos'].max()
tweet_pos_mid = (tweet_pos_min + tweet_pos_max) / 2

tweet_neg_min = data['TweetNeg'].min()
tweet_neg_max = data['TweetNeg'].max()
tweet_neg_mid = (tweet_neg_min + tweet_neg_max) / 2

# Fuzzify TweetPos
def fuzzify_pos(score):
    return {
        'Low_Pos': triangular_membership(score, tweet_pos_min, tweet_pos_min, tweet_pos_mid),
        'Medium_Pos': triangular_membership(score, tweet_pos_min, tweet_pos_mid, tweet_pos_max),
        'High_Pos': triangular_membership(score, tweet_pos_mid, tweet_pos_max, tweet_pos_max)
    }

# Fuzzify TweetNeg
def fuzzify_neg(score):
    return {
        'Low_Neg': triangular_membership(score, tweet_neg_min, tweet_neg_min, tweet_neg_mid),
        'Medium_Neg': triangular_membership(score, tweet_neg_min, tweet_neg_mid, tweet_neg_max),
        'High_Neg': triangular_membership(score, tweet_neg_mid, tweet_neg_max, tweet_neg_max)
    }

# Apply fuzzification to TweetPos and TweetNeg
fuzzified_pos = data['TweetPos'].apply(fuzzify_pos)
fuzzified_neg = data['TweetNeg'].apply(fuzzify_neg)

# Convert fuzzified results to DataFrame and concatenate with the original data
fuzzified_pos_df = pd.DataFrame(fuzzified_pos.tolist())
fuzzified_neg_df = pd.DataFrame(fuzzified_neg.tolist())

# Concatenate the fuzzified columns to the original dataset
data = pd.concat([data, fuzzified_pos_df.add_prefix('Fuzzy_'), fuzzified_neg_df.add_prefix('Fuzzy_')], axis=1)

# Save the fuzzified dataset
data.to_csv('dataset_fuzzified.csv', index=False)

# Display the fuzzified dataset
print(data.head())
