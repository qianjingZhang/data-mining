# data-mining
data cleaning

the code is used to solve the problems about data missing, outlier and duplicated data and the dataset is Newstudents.xlsx

About data missing,

based on the ratio of missing data, if it is above 40%, we choose to delete it directly; otherwise, we will use guassain distribution to produce data.

About outlier,

we use LOF based on two-level, feature-level and item-level

About duplicated data,

we directly use ID as the key. Of course, we can also extract information from some features and then integrate them into key value
# data-mining
data cleaning
