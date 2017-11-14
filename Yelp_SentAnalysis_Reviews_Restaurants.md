
                            Yelp Data - Exploratory Analysis of Restaurants in US 
Objective of the project 
This is a code file to analyze Yelp Data on business and reviews. This small project is to explore and analyze two files - 
business data and review data - and then to find if there exist relationship between stars and reviews by doing sentiment analysis. 

Description of the files analyzed
Business data contains information about business name and type, location, hours of operation, geo-spatial variables, features or attributes of businesses, number of reviews and average star or rating. Business data has information about 156,000 business entities 

Review data contains reviews (text), stars given by reviewers, business id that received reviews in Yelp, usefulness of reviews, review date etc. Review data contains 4.7 million reviews. 

The project focuses only on restaurants. This subset of data was extracted using category file. We took a sample of first 300,000 reviews. Computer memory issue restricted us to take this sample size. 

The below codes, second file in this project, focuses mainly on sentiment analysis

```python
# Import dependencies 
import argparse
import collections
import csv
import json
import pandas as pd
from itertools import islice

import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Load Busienss Data 
business= pd.read_json('business.json', orient= "records", lines=True)
business.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>attributes</th>
      <th>business_id</th>
      <th>categories</th>
      <th>city</th>
      <th>hours</th>
      <th>is_open</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighborhood</th>
      <th>postal_code</th>
      <th>review_count</th>
      <th>stars</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>691 Richmond Rd</td>
      <td>{'RestaurantsPriceRange2': 2, 'BusinessParking...</td>
      <td>YDf95gJZaq05wvo7hTQbbQ</td>
      <td>[Shopping, Shopping Centers]</td>
      <td>Richmond Heights</td>
      <td>{'Monday': '10:00-21:00', 'Tuesday': '10:00-21...</td>
      <td>1</td>
      <td>41.541716</td>
      <td>-81.493116</td>
      <td>Richmond Town Square</td>
      <td></td>
      <td>44143</td>
      <td>17</td>
      <td>2.0</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2824 Milton Rd</td>
      <td>{'GoodForMeal': {'dessert': False, 'latenight'...</td>
      <td>mLwM-h2YhXl2NCgdS84_Bw</td>
      <td>[Food, Soul Food, Convenience Stores, Restaura...</td>
      <td>Charlotte</td>
      <td>{'Monday': '10:00-22:00', 'Tuesday': '10:00-22...</td>
      <td>0</td>
      <td>35.236870</td>
      <td>-80.741976</td>
      <td>South Florida Style Chicken &amp; Ribs</td>
      <td>Eastland</td>
      <td>28215</td>
      <td>4</td>
      <td>4.5</td>
      <td>NC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>337 Danforth Avenue</td>
      <td>{'BusinessParking': {'garage': False, 'street'...</td>
      <td>v2WhjAB3PIBA8J8VxG3wEg</td>
      <td>[Food, Coffee &amp; Tea]</td>
      <td>Toronto</td>
      <td>{'Monday': '10:00-19:00', 'Tuesday': '10:00-19...</td>
      <td>0</td>
      <td>43.677126</td>
      <td>-79.353285</td>
      <td>The Tea Emporium</td>
      <td>Riverdale</td>
      <td>M4K 1N7</td>
      <td>7</td>
      <td>4.5</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7702 E Doubletree Ranch Rd, Ste 300</td>
      <td>{}</td>
      <td>CVtCbSB1zUcUWg-9TNGTuQ</td>
      <td>[Professional Services, Matchmakers]</td>
      <td>Scottsdale</td>
      <td>{'Friday': '9:00-17:00', 'Tuesday': '9:00-17:0...</td>
      <td>1</td>
      <td>33.565082</td>
      <td>-111.916400</td>
      <td>TRUmatch</td>
      <td></td>
      <td>85258</td>
      <td>3</td>
      <td>3.0</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4719 N 20Th St</td>
      <td>{'RestaurantsTableService': False, 'GoodForMea...</td>
      <td>duHFBe87uNSXImQmvBh87Q</td>
      <td>[Sandwiches, Restaurants]</td>
      <td>Phoenix</td>
      <td>{}</td>
      <td>0</td>
      <td>33.505928</td>
      <td>-112.038847</td>
      <td>Blimpie</td>
      <td></td>
      <td>85016</td>
      <td>10</td>
      <td>4.5</td>
      <td>AZ</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select only restaurants from business data file
restaurants = business[business['categories'].apply(lambda x: 'Restaurants' in x)]
restaurants.head()
len(restaurants)
```




    51613




```python
# Check which cities has largest number of restaurants. Later analysis is based on top five US cities,
# namely Las Vegas, Phoenix, Charlotte, Pittsburgh and Cleveland

restaurants['city'].value_counts().head(10)
```




    Toronto        6750
    Las Vegas      5682
    Phoenix        3515
    Montréal       3101
    Charlotte      2327
    Pittsburgh     2089
    Edinburgh      1437
    Scottsdale     1414
    Cleveland      1292
    Mississauga    1228
    Name: city, dtype: int64




```python
# Descriptive statistics of number of reviews of the complete restaurants data 
restaurants['review_count'].describe()
```




    count    51613.000000
    mean        56.720826
    std        144.264792
    min          3.000000
    25%          7.000000
    50%         18.000000
    75%         52.000000
    max       6979.000000
    Name: review_count, dtype: float64




```python
# Descriptive statistics of stars/rating of the complete restaurants data 
restaurants['stars'].describe()
```




    count    51613.000000
    mean         3.461105
    std          0.783030
    min          1.000000
    25%          3.000000
    50%          3.500000
    75%          4.000000
    max          5.000000
    Name: stars, dtype: float64




```python
# Frequency distribution of stars of the restaurants of the full dataset 
restaurants['stars'].value_counts()
```




    4.0    12922
    3.5    12747
    3.0     9434
    4.5     6064
    2.5     5108
    2.0     2689
    5.0     1366
    1.5      969
    1.0      314
    Name: stars, dtype: int64




```python
# Split the restaurants into active vs closed using the variable 'is_opn'
restaurants_active = restaurants[restaurants['is_open']==1]

# There is 38,657 restaurants found open 
len(restaurants_active)
```




    38657




```python
# closed restaurants 
restaurants_closed = restaurants[restaurants['is_open']==0]

# 12,956 resturants found closed 
len(restaurants_closed)
```




    12956




```python
# Load Review Data File with 300,000 rows/observations 

reviews = ''

with open('review.json', encoding="utf8") as f:
    for line in f.readlines()[0:300000]:
        reviews += line

review = pd.read_json(reviews, orient= "records", lines=True)
```


```python
# Check the review data set 
review.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2016-07-12</td>
      <td>0</td>
      <td>VfBHSwC5Vz_pbFluy07i9Q</td>
      <td>5</td>
      <td>My girlfriend and I stayed here for 3 nights a...</td>
      <td>0</td>
      <td>cjpdDjZyprfyDG3RlkVG3w</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2016-10-02</td>
      <td>0</td>
      <td>3zRpneRKDsOPq92tq7ybAA</td>
      <td>3</td>
      <td>If you need an inexpensive place to stay for a...</td>
      <td>0</td>
      <td>bjTcT8Ty4cJZhEOEo01FGA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2015-09-17</td>
      <td>0</td>
      <td>ne5WhI1jUFOcRn-b-gAzHA</td>
      <td>3</td>
      <td>Mittlerweile gibt es in Edinburgh zwei Ableger...</td>
      <td>0</td>
      <td>AXgRULmWcME7J6Ix3I--ww</td>
    </tr>
    <tr>
      <th>3</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2016-08-21</td>
      <td>0</td>
      <td>llmdwOgDReucVoWEry61Lw</td>
      <td>4</td>
      <td>Location is everything and this hotel has it! ...</td>
      <td>0</td>
      <td>oU2SSOmsp_A8JYI7Z2JJ5w</td>
    </tr>
    <tr>
      <th>4</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2013-11-20</td>
      <td>0</td>
      <td>DuffS87NaSMDmIfluvT83g</td>
      <td>5</td>
      <td>gute lage im stadtzentrum. shoppingmeile und s...</td>
      <td>0</td>
      <td>0xtbPEna2Kei11vsU-U2Mw</td>
    </tr>
  </tbody>
</table>
</div>




```python
# saving this review file to csv 
review.to_csv('review.csv')
```


```python
# Merging business data with review data 
yelp_complete_df = pd.merge(business, review, on ='business_id')
```


```python
# Merging restaurants data that where restaurants are active with review data
restaurants_active_review = pd.merge(restaurants_active, review, on ='business_id')

# Merged file that 142,774 reviews
len(restaurants_active_review)
```




    142774




```python
# Saving restaurants active file to csv 
restaurants_active_review.to_csv('restaurants_active_review.csv')
```


```python
# Merging restaurants data that where restaurants are closed with review data
restaurants_closed_review = pd.merge(restaurants_closed, review, on ='business_id')

# Merged file that 31,494 reviews
len(restaurants_closed_review)
```




    31494




```python
# Saving restaurants closed file to csv 
restaurants_closed_review.to_csv('restaurants_closed_review.csv')
```


```python
# select active restaurants of the following five cities 
city_names = ('Las Vegas', 'Pittsburgh', 'Phoenix', 'Charlotte', 'Cleveland')
restaurants_fiveCity = restaurants_active_review.loc[restaurants_active_review['city'].isin(city_names)]
len(restaurants_fiveCity)
```




    72586




```python
# Conduct sentiment analysis of the reviews of the restaurants in the five cities 
# Create empty list for compound, positive, negative and neutral scores for each business id 
compound = []
pos = []
neu =[]
neg = []
business_id = []
text = []
```


```python
# Running loop operation to populate the lists of compound, positive, negative and neutral scores 
for index, row in restaurants_fiveCity.iterrows():
    business_id.append(row['business_id'])
    text.append(row['text'])
    compound.append(analyzer.polarity_scores(row['text'])['compound'])
    pos.append(analyzer.polarity_scores(row['text'])['pos'])
    neu.append(analyzer.polarity_scores(row['text'])['neu'])
    neg.append(analyzer.polarity_scores(row['text'])['neg'])  
```


```python
# Create a dataframe of sentiment scores 
sentiments = pd.DataFrame({'business_id': business_id,
                           'text': text,
                           'Compound': compound,
                            'Positive': pos,
                            'Neutral': neu,
                            'Negative': neg})
len(sentiments)
```




    72586




```python
# Merge the sentiment data with restaurants data (active in five cities)
rest_active_sentiment = pd.merge(restaurants_fiveCity, sentiments, on =['business_id', 'text'])
```


```python
# Identify the columns that are of less interest and can be dropped
rest_active_sentiment.columns
```




    Index(['address', 'attributes', 'business_id', 'categories', 'city', 'hours',
           'is_open', 'latitude', 'longitude', 'name', 'neighborhood',
           'postal_code', 'review_count', 'stars_x', 'state', 'cool', 'date',
           'funny', 'review_id', 'stars_y', 'text', 'useful', 'user_id',
           'Compound', 'Negative', 'Neutral', 'Positive'],
          dtype='object')




```python
# Working with a cleaner and small set of data that contains sentiment scores, review, rating by business id
columns = ['review_id', 'address', 'attributes', 'hours', 'is_open', 'latitude', 'longitude', 'name', 
           'neighborhood', 'postal_code', 'date', 'user_id']
rest_active_sentiment.drop(columns, inplace=True, axis=1)
```


```python
# Renaming the two merged variables
rest_active_sentiment.rename(columns={'stars_x': 'average_star', 'stars_y': 'starByreviewer'}, inplace=True)
```


```python

closedRestaurants_fiveCity = restaurants_closed_review.loc[restaurants_closed_review['city'].isin(city_names)]
len(closedRestaurants_fiveCity)
```




    14529




```python
# Conduct sentiment analysis of the reviews of the closed restaurants in the five cities 
# Create empty list for compound, positive, negative and neutral scores for each business id 
cl_compound = []
cl_pos = []
cl_neu =[]
cl_neg = []
business_id = []
text = []
```


```python
# Running loop operation to populate the lists of compound, positive, negative and neutral scores 
for index, row in closedRestaurants_fiveCity.iterrows():
    business_id.append(row['business_id'])
    text.append(row['text'])
    cl_compound.append(analyzer.polarity_scores(row['text'])['compound'])
    cl_pos.append(analyzer.polarity_scores(row['text'])['pos'])
    cl_neu.append(analyzer.polarity_scores(row['text'])['neu'])
    cl_neg.append(analyzer.polarity_scores(row['text'])['neg'])  
```


```python
# Create a dataframe of sentiment scores 
sentiments_closed = pd.DataFrame({'business_id': business_id,
                           'text': text,
                           'Compound': cl_compound,
                            'Positive': cl_pos,
                            'Neutral': cl_neu,
                            'Negative': cl_neg})
```


```python
# Merge the sentiment data with restaurants data (closed in five cities)
rest_closed_sentiment = pd.merge(closedRestaurants_fiveCity, sentiments_closed, on =['business_id', 'text'])
```


```python
# Renaming the two merged variables
rest_closed_sentiment.rename(columns={'stars_x': 'average_star', 'stars_y': 'starByreviewer'}, inplace=True)
```


```python
# Working with a cleaner and small set of data that contains sentiment scores, review, rating by business id
columns = ['review_id', 'address', 'attributes', 'hours', 'is_open', 'latitude', 'longitude', 'name', 
           'neighborhood', 'postal_code', 'date', 'user_id']
rest_closed_sentiment.drop(columns, inplace=True, axis=1)
```


```python
# Keeping only selected variables from the active restaurants-sentiment file 
rest_active_select = rest_active_sentiment.loc[:,('business_id', 'categories', 'city', 'review_count', 'average_star',
                                                  'Compound', 'Negative', 'Positive', 'Neutral')]
```


```python
# Explore how many chinese and mexican restaurants in these five cities 
rest_active_select['is_chinese'] = rest_active_select['categories'].apply(lambda x: 'Chinese' in x)
rest_active_select['is_mexican'] = rest_active_select['categories'].apply(lambda x: 'Mexican' in x)
```


```python
# Keeping only selected variables from the active restaurants-sentiment file 
rest_closed_select = rest_closed_sentiment.loc[:,('business_id', 'categories', 'city', 'review_count', 'average_star',
                                                  'Compound', 'Negative', 'Positive', 'Neutral')]
```


```python
rest_closed_select['is_chinese'] = rest_closed_select['categories'].apply(lambda x: 'Chinese' in x)
rest_closed_select['is_mexican'] = rest_closed_select['categories'].apply(lambda x: 'Mexican' in x)
```


```python
# Compute mean sentiment scores by business id for active restaurants 
rest_active_agg = pd.DataFrame(rest_active_select.groupby('business_id')['Compound'].mean())
rest_active_agg['city'] = rest_active_select.groupby('business_id')['city'].unique()
rest_active_agg['review_count'] = rest_active_select.groupby('business_id')['review_count'].mean()
rest_active_agg['average_star'] = rest_active_select.groupby('business_id')['average_star'].mean()
rest_active_agg['avg_negative'] = rest_active_select.groupby('business_id')['Negative'].mean()
rest_active_agg['avg_positive'] = rest_active_select.groupby('business_id')['Positive'].mean()
rest_active_agg['avg_neutral'] = rest_active_select.groupby('business_id')['Neutral'].mean()
rest_active_agg['is_chinese'] = rest_active_select.groupby('business_id')['is_chinese'].unique()
rest_active_agg['is_mexican'] = rest_active_select.groupby('business_id')['is_mexican'].unique()
rest_active_agg.reset_index(inplace=True)
```


```python
# Do further cleaning of the aggregated data (aggregation above creates list inside dataframe, so needed to make it regular colum)
rest_active_agg['city'] = rest_active_agg['city'].apply(lambda x: x[0])
rest_active_agg['is_chinese'] = rest_active_agg['is_chinese'].apply(lambda x: x[0])
rest_active_agg['is_mexican'] = rest_active_agg['is_mexican'].apply(lambda x: x[0])
```


```python
# Renaming compound variable to average compound 
rest_active_agg.rename(columns={'Compound': 'avg_compound'}, inplace=True)
```


```python
# Only a small fraction of the active restaurants are chinese
rest_active_agg['is_chinese'].value_counts()
```




    False    718
    True      48
    Name: is_chinese, dtype: int64




```python
# Compute mean sentiment scores by business id for active restaurants 
rest_closed_agg = pd.DataFrame(rest_closed_select.groupby('business_id')['Compound'].mean())
rest_closed_agg['city'] = rest_closed_select.groupby('business_id')['city'].unique()
rest_closed_agg['review_count'] = rest_closed_select.groupby('business_id')['review_count'].mean()
rest_closed_agg['average_star'] = rest_closed_select.groupby('business_id')['average_star'].mean()
rest_closed_agg['avg_negative'] = rest_closed_select.groupby('business_id')['Negative'].mean()
rest_closed_agg['avg_positive'] = rest_closed_select.groupby('business_id')['Positive'].mean()
rest_closed_agg['avg_neutral'] = rest_closed_select.groupby('business_id')['Neutral'].mean()
rest_closed_agg['is_chinese'] = rest_closed_select.groupby('business_id')['is_chinese'].unique()
rest_closed_agg['is_mexican'] = rest_closed_select.groupby('business_id')['is_mexican'].unique()
rest_closed_agg.reset_index(inplace=True)
```


```python
# Do further cleaning of the aggregated data (aggregation above creates list inside dataframe, so needed to make it regular colum)
rest_closed_agg['city'] = rest_closed_agg['city'].apply(lambda x: x[0])
rest_closed_agg['is_chinese'] = rest_closed_agg['is_chinese'].apply(lambda x: x[0])
rest_closed_agg['is_mexican'] = rest_closed_agg['is_mexican'].apply(lambda x: x[0])
```


```python
# Renaming compound variable to average compound 
rest_closed_agg.rename(columns={'Compound': 'avg_compound'}, inplace=True)
```


```python
# Only a small fraction of the closed restaurants are mexican 
rest_closed_agg['is_mexican'].value_counts()
```




    False    372
    True      46
    Name: is_mexican, dtype: int64




```python
# Create operating status column for aggregated restaurant data files
rest_active_agg['active']= 'active'
rest_closed_agg['active']= 'closed'
```


```python
# Create a merged data file for the analysis and graphing 
all_types = [rest_active_agg, rest_closed_agg]
rest_all_types = pd.concat(all_types)
columns = ['is_chinese', 'is_mexican']
rest_all_types.drop(columns, inplace=True, axis=1)

# A clean file with active and closed restaurants in five cities that has sentiment scores, number of reviews, 
# average number of  and average star by business id. The length of this file is 1184 
len(rest_all_types)
```




    1184




```python
# Descriptive statistics of star/rating in the subset of restaurants data (of only five cities)
rest_all_types['average_star'].describe()
```




    count    1184.000000
    mean        3.413007
    std         0.807844
    min         1.000000
    25%         3.000000
    50%         3.500000
    75%         4.000000
    max         5.000000
    Name: average_star, dtype: float64




```python
# Descriptive statistics of average compound score in the subset of restaurants data (of only five cities)
rest_all_types['avg_compound'].describe()
```




    count    1184.000000
    mean        0.587731
    std         0.280045
    min        -0.799100
    25%         0.455811
    50%         0.665918
    75%         0.787856
    max         0.995833
    Name: avg_compound, dtype: float64




```python
# Does rating increase with the number of reviews? The first scatter plot of number of reviews against  average star. 

g = (sns.lmplot('average_star', 'review_count', data =rest_all_types, fit_reg=False).set_axis_labels("Average Star", 
            "Number of reviews"))
g.fig.set_size_inches(10,7)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Does rating increase with the number of reviews?", size =15)
g.savefig('Fig10-ReviewVSStar.png')
plt.show()
```


![png](output_50_0.png)



```python
# Distribution of stars - Open vs Closed Restaurants in Five Cities

g = (sns.factorplot(x="city", y="average_star",hue="active", col = 'active', data=rest_all_types, kind="box").set_axis_labels(
    "City", "Stars/Rating"))
g.fig.set_size_inches(13,5)
plt.subplots_adjust(top=0.85)
g.fig.suptitle("Distribution of stars - Open vs Closed Restaurants in Five Cities", 
               size =15)
g.savefig('Fig1-BoxPlot of Stars.png')
plt.show()
```


![png](output_51_0.png)

Most cities have similar distribution in terms of rating, except Phoenix. Rating ranges from 3 to 4 in most cities in the case of open restaurants. 

Average rating is 3.5 across all cities in open and closed restaurants

```python
# Facted graph: Box plot of average compound score-open vs closed restaurants-in five cities

g = (sns.factorplot(x="city", y="avg_compound",hue="active", col = 'active', data=rest_all_types, kind="box").set_axis_labels(
    "City", "Average Compound Score"))
g.fig.set_size_inches(13,5)
plt.subplots_adjust(top=0.85)
g.fig.suptitle("Average Compound Score-Open vs Closed Restaurants-in five cities", 
               size =15)
g.savefig('Fig2-BoxPlot-AvgCompound.png')
plt.show()
```


![png](output_53_0.png)

It is observed that closed restaurants have received higher compound score than the ones in open category for most cities in this list. 

The city Pittsburgh is found to receive highest compound score. The restaurant goers seem very positive about the restaurants in Pittsburg. Among the closed restaurants, Charlotte has restaurants with higher range of compound score. 

```python
# Sentiments expressed by reviewers (measured by average compound score, plotted against average star of the restaurants 

g = sns.FacetGrid(rest_all_types, col = 'active', hue= 'active')
g = (g.map(sns.regplot, 'average_star', 'avg_compound', order=2, ci=None).set_axis_labels("Average Star", 
            "Average Compound Score"))
g.fig.set_size_inches(15,9)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Sentiment against Rating of Restaurants - Open vs Closed", size =15)
g.savefig('Fig3-AvgCompoundVsStar-ActClose_City.png')
plt.show()
```


![png](output_55_0.png)

The abvoe scatter plot with a higher order linear regression fit reveals that as the restaurants receive higher ratings, they also received higher average compound scores. The compound score reaches peak at rating of 3.5 for both active and closed restaurants.

The relationship is more linear in the case of open restaurants as is in the case of closed restaurants. 

```python
# Facted graph: scatter plot of average compound score vs average star of the open and closed restaurants in five cities 

g = sns.FacetGrid(rest_all_types, col = 'city', row = 'active', hue= 'active')
g = (g.map(sns.regplot, 'average_star', 'avg_compound', order=2, ci=None).set_axis_labels("Average Star", 
            "Average Compound Score"))
g.fig.set_size_inches(15,9)
#g.despine(offset=10)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Sentiment against Rating of Restaurants - Open vs Closed -in five major cities", size =15)
g.savefig('Fig4-AvgCompoundVsStar-ActClose_City.png')
plt.show()
```


![png](output_57_0.png)

As seen in the previous graphs, the plots for each city confirms the linear relationship between rating and average compound score. 

For open restaurants, the line is more upward slopping. However, for closed restaurants the line is of polynomial type. 


```python
# Sentiment expressed by reviewers vs review count- Open vs Closed Restaurants in five cities

g = sns.FacetGrid(rest_all_types, col = 'city', row = 'active', hue = 'active')
g = (g.map(plt.scatter, "review_count", "avg_compound").set_axis_labels("Review Count", "Average Compound Score"))
g.fig.set_size_inches(15,9)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Sentiment expressed by reviewers vs review count- Open vs Closed Restaurants in five cities", 
               size =15)
g.savefig('Fig5-AvgCompound-Vs-Review_AllTypes.png')
plt.show()
```


![png](output_59_0.png)



```python
# Density plot of Sentiment Score - Open vs Closed Restaurants in Five Cities

g = sns.FacetGrid(rest_all_types, col="city", row = 'active', hue= 'active')
g = (g.map(sns.distplot, "avg_compound").set_xlabels("Average Compound Score"))
g.fig.set_size_inches(15,9)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Density plot of Sentiment Score - Open vs Closed Restaurants in Five Cities", 
               size =15)
g.savefig('Fig6-DensityPlot-AvgCompound-AllTypes.png')
plt.show()
```


![png](output_60_0.png)



```python
# Scatter plot of Rating vs Number of Reviews - Open and Closed Restaurants

g = sns.FacetGrid(rest_all_types, col = 'active', hue= 'active')
g = (g.map(plt.scatter, 'review_count', 'average_star').set_axis_labels("Number of Reviews", 
            "Average Star"))
g.fig.set_size_inches(15,7)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Rating vs Number of Reviews - Open and Closed Restaurants", size =18)
g.savefig('Fig7-AvgCompoundVsStar-ActClose_City.png')
plt.show()
```


![png](output_61_0.png)

The above scatter plot doesn't reveal any relationship between average star and number of reviews. 
# Final Comments
The sentiment analysis and the plots against rating - at overall level as well for the cities - are consistent with the rating, i.e. increased compound score is associated with higher rating.

Based on the sentiment analysis, we can infer that people are less likely to give a 1 or 5 star review. 

There is no discernable difference in results of reviews between open and closed restaurants. 
# References

1. Link for the source of the Yelp data 
https://www.yelp.com/dataset

2. Seaborn tutorial on scatter plot 
https://chrisalbon.com/python/seaborn_scatterplot.html

3. Link to seaborn library on faceted graph 
https://seaborn.pydata.org/tutorial/axis_grids.html

4. Link to seaborn for linear fitting to scatter plot 
https://seaborn.pydata.org/generated/seaborn.regplot.html

5. Few stackoverflow links to nice plotting using seaborn library 
https://stackoverflow.com/questions/29637150/scatterplot-without-linear-fit-in-seaborn

6. Link to advance funcationality of seaborn library 
https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6
