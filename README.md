# Detection of Poisonous and Edible Mushrooms by Classification
## 1.INTRODUCTION
 How do we tell whether a mushroom is poisonous or edible by looking at the data in a dataset? 
The appearance of mushrooms gives us a lot of information about them.For example, the shape and color of the caps and stalks of mushrooms, their odor, and the shape of their roots give us information about their species.Collecting all these features of mushrooms in one dataset makes them easy to analyze.Based on the information in this dataset, we can use the classification technique (which is a data mining method), to understand which mushroom is poisonous or edible.In this project, I determined whether a mushroom is poisonous or edible by using the classification technique that I mentioned above. 
## 2.MATERIALS AND METHODS
### Dataset Name: Mushroom Classification
### Dataset Link: https://www.kaggle.com/uciml/mushroom-classification
### Note:
Mushroom Classification dataset was originally donated to the UCI Machine Learning repository (27 April 1987). </br>
*For more information about dataset visit this site: https://archive.ics.uci.edu/ml/datasets/Mushroom* 
### Content
This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one.  </br>
## METHOD
In this project, I used a predictive data mining method ,the classification technique.

### Predictive Data Mining:
The main goal of this mining is to say something about future results not of current behaviour. It uses the supervised learning functions which are used to predict the target value. The methods come under this type of mining category are called classification, time-series analysis and regression. Modelling of data is the necessity of the predictive analysis, and it works by utilizing a few variables of the present to predict the future not known data values for other variables.

### What is Classification in Data Mining ? 
Classification in data mining is a common technique that separates data points into different classes. It allows you to organize data sets of all sorts, including complex and large datasets as well as small and simple ones. 
It primarily involves using algorithms that you can easily modify to improve the data quality. This is a big reason why supervised learning is particularly common with classification in techniques in data mining. The primary goal of classification is to connect a variable of interest with the required variables. The variable of interest should be of qualitative type. 
The algorithm establishes the link between the variables for prediction. The algorithm you use for classification in data mining is called the classifier, and observations you make through the same are called the instances. You use classification techniques in data mining when you have to work with qualitative variables. 
There are multiple types of classification algorithms, each with its unique functionality and application. All of those algorithms are used to extract data from a dataset. Which application you use for a particular task depends on the goal of the task and the kind of data you need to extract. 

## 4.CODES

*I used Python and Google Colaboratory for coding.*

-Defining Libraries
```
import io 						//path for accessing file
import numpy as np 				//for linear algebra
import pandas as pd 				//for data processing
import matplotlib.pyplot as plt 			//ploting library
import seaborn as sns				//ploting library
```
-Uploading .csv file (dataset)
```
from google.colab import files         	
uploaded = files.upload()								
data= pd.read_csv(io.BytesIO(uploaded['mushrooms.csv'])) 	
data							// looking data for the first time
```
-Controlling missing values
```
data.isnull().sum()
```
-Column by column value distributions
```
columns = data.columns.to_list() 			//using list() constructor
print("*column by column data distributions*\n")
for col in columns:
print(col,"\n",data[col].value_counts(),"\n\n")	//printing the list 
-Data visualization of my Target (class: edible or poisonous) by using Pyplot and seaborn
total = float(len(data[columns[0]]))
plt.figure(figsize=(4,4))
sns.set(style="dark")
i = sns.countplot(data[columns[0]])
for j in i.patches:
    height = j.get_height()
    i.text(j.get_x()+j.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center")
plt.title("Target ", fontsize = 15)
plt.show()
```
-Data visualization of dataset’s columns (attributes) by using Pyplot and seaborn. And attribute percentages 
```
for col in columns[1:]:
    plt.figure(figsize=(7,4))				
    sns.countplot(x=col , data=data ,palette='cubehelix')
    plt.title(col, fontsize=15)
    plt.show()
    print("% of total:")
print(round((data[col].value_counts()/data.shape[0]),4)*100)
/* In last row I used Series.value_counts() for returning  my dataset's attributes containing counts of unique values.Also I used Series.shape for returning a tuple of the shape of my underlying data.Finally, I calculated it as a percentage and converted it to float.
*/
```
-Multivariate plots (with poisonous / edible) by using Pyplot and seaborn.
```
for col in columns[1:]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=col ,hue='class', data=data ,palette='cubehelix')
    plt.xlabel(col, fontsize=15)
    plt.legend(loc='upper right')
```
-Lastly, I showed all the data in pivot table
```
[ pd.pivot_table(data, index=[col,"class"], aggfunc = {col:np.count_nonzero}) for col in columns[1:]]
```
## 5.RESULTS
- cap-shape - Most of the knobbed mushrooms in our dataset are poisonous.
- cap-surface - Most fibrous cap surface are edible.
- cap-color - Most white cap colored mushrooms are edible while most yellow cap colored mushrooms are poisonous.
- bruises - Bruised mushrooms are usually edible while unbruised ones are usually the opposite.
- odor - No smell mushrooms are mostly edible by a wide margin while all foul smell mushrooms are poisonous.
- gill-attachment - The attached gills are almost always edible.
- gill-spacing - The crowded gills are almost always edible.
- gill-size - The narrow gill sized mushrooms are almost always poisonous.
- gill-color - The buff gill colored mushrooms are always poisonous.
- stalk-shape - insignificant difference between each value in terms of poisonous or edible.
- stalk-root - Mushrooms with missing data of stalk roots are usually poisonous.
- stalk-surface-above-ring - The silky mushrooms are usually poisonous, smooth are usually edible.
- stalk-surface-below-ring - About the same as stalk-surface-above-ring.
- stalk-color-above-ring - The white stalk colored mushrooms are usually edible, pink ones are mostly poisonous.
- stalk-color-below-ring - About the same as stalk-color-above-ring.
- veil-type - All veil type of the mushroom's are partial so this column is pretty much useless in our analysis.
- veil-color - Almost all of the mushroom's veil color are white (97.54%) so this column is pretty much useless in our analysis.
- ring-number - Almost all of the mushroom's ring number amount are one (92.17%) so this column is pretty much useless in our analysis.
- ring-type - The pendant ring typed mushrooms are mostly edible,evanescent are mostly poisonous and large ring types are all poisonous.
- spore-print-color - The brown and black ones are almost entirely edible while white and chocolate (20.09%) are mostly poisonous.
- population - Mushrooms with a several population are mostly poisonous.
- habitat - The woods or grasses grown mushrooms are mostly edible.
## REFERENCES
https://www.kaggle.com/uciml/mushroom-classification </br>
https://archive.ics.uci.edu/ml/datasets/Mushroom </br>
https://bilgisayarkavramlari.com/2013/03/31/siniflandirma-classification/ </br>
https://www.geeksforgeeks.org/basic-concept-classification-data-mining/ </br>
https://www.upgrad.com/blog/classification-in-data-mining/ </br>
https://www.geeksforgeeks.org/difference-between-descriptive-and-predictive-data-mining/ </br>
https://matplotlib.org/stable/tutorials/introductory/pyplot.html </br>
https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html </br>
https://www.kaggle.com/yonatanrabinovich/mushroom-classification-project </br>
https://pandas.pydata.org/docs/reference/api/pandas.Series.shape.html </br>
https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html </br>
https://www.kaggle.com/yonatanrabinovich/mushroom-classification-project </br>
https://web.itu.edu.tr/~sgunduz/courses/verimaden/slides/d3.pdf </br>
https://www.youtube.com/watch?v=SIU_GWaEiIs </br>
https://stackoverflow.com/questions/48485255/how-can-access-uploaded-file-in-google-colab </br>
https://seaborn.pydata.org/tutorial/color_palettes.html </br>
https://stackoverflow.com/questions/41384040/subplot-for-seaborn-boxplot </br>
https://seaborn.pydata.org/generated/seaborn.countplot.html </br>
https://stackoverflow.com/questions/34193862/pandas-pivot-table-list-of-aggfunc </br>
