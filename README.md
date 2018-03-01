# Machine Learning Big Data Starter


This artifact is further progression on my previous work [ml-starter](https://github.com/ERS-HCL/ml-starter) which is standalone version of spark-mllib's ALS demonstration based using local CSV files.

### Note : Please read the above in detail before proceeding further. 

To go one step first further I have replaced the file system layer with MongoDB.

MongoDB provide a spark-mongo connector that wrap standard Java/Scala connector with spark interpolatable data format/API's.  


# Getting Started

Apart from the spark core API's you would need following dependency to connect to MongoDB server. 

```

<dependency>
	<groupId>org.mongodb.spark</groupId>
	<artifactId>mongo-spark-connector_2.11</artifactId>
	<version>2.2.1</version>
</dependency>

 ```

# Preparing the data

In current scenario i.e MongoDB instead of creating files we need same data in BSON format in collections. 

Sales Orders :


UserId  UserName       ProductId     ProductName   Rate  Quantity  Amount

1       User 1         1             Product 1     10    5         50

1       User 1         2             Product 2     20    10        200

1       User 1         3             Product 3     10    15        150

2       User 2         1             Product 1     10    5         50

2       User 2         2             Product 2     20    20        400

2       User 2         4             Product 4     10    15        150


Sales Leads :


UserId  UserName      ProductId     ProductName

1       User 1        4             Product 4

1       User 1        5             Product 5

2       User 2        3             Product 3

2       User 2        6             Product 6


We need to predict/recommend most relevant Product for both the user based onto their past order history. Here we can see Both User 1 and User 2 ordered Product 1 and Product 2, also they have ordered one item separately. 

Now we predicting their rating for alternate product and one new product.

# Implementaion

## #1

Our first step would be making db connection using MongoDB specific properties.

```

SparkConf conf = new SparkConf().//
	setAppName("rnd").//
	setMaster("local").//
	set("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/sparkdb.myCollection").//
	set("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/sparkdb.myCollection");

```

## #2

Now you can read training model via JavaMongoRDD API and convert it to Rating format using JavaRDD API.

```

private static JavaMongoRDD<Document> getJavaMongoRDD(JavaSparkContext jsc, String collName) {

	Map<String, String> readOverrides = new HashMap<String, String>();
	readOverrides.put("collection", collName);
	readOverrides.put("readPreference.name", "secondaryPreferred");
	ReadConfig readConfig = ReadConfig.create(jsc).withOptions(readOverrides);

	JavaMongoRDD<Document> mongoRDD = MongoSpark.load(jsc, readConfig);
	return mongoRDD;
}

JavaMongoRDD<Document> salesOrdersRDD = getJavaMongoRDD(jsc, "SalesOrders");


// Map file to Ratings(user,item,rating) tuples
JavaRDD<Rating> ratings = salesOrdersRDD.map(new Function<Document, Rating>() {
	public Rating call(Document d) {
		return new Rating(d.getInteger("userCode"), d.getInteger("productCode"), ((Number) d.get("amount")).doubleValue());
	}
});

  
```

## #3

Next step is to train the Matrix Factorization model using ALS algorithm.

```
MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations); 
```

## #4

Now you can load the Sales Lead file and convert it to Tupple format.

```
JavaMongoRDD<Document> salesLeadsRDD = getJavaMongoRDD(jsc, "SalesLeads");

// Create user-item tuples from ratings
JavaRDD<Tuple2<Object, Object>> userProducts = salesLeadsRDD.map(new Function<Document, Tuple2<Object, Object>>() {
	public Tuple2<Object, Object> call(Document d) {
		return new Tuple2<Object, Object>(d.getInteger("userCode"), d.getInteger("productCode"));
	}
});

```

## #5

Finally we can predict the future rating using simple API.

```
// Predict the ratings of the products not rated by user 
JavaRDD<Rating> recomondations = model.predict(userProducts.rdd()).toJavaRDD().distinct();

```

## #6

Optionally you can sort the output using basic pipeline operation

```
// Sort the recommendations by rating in descending order 
recomondations = recomondations.sortBy(new Function<Rating, Double>() {  
  @Override  
  public Double call(Rating v1) throws Exception {   
    return v1.rating();  
  }
 }, false, 1);

```

## #7

Now you display your result using basic JavaRDD API.

```
// Print the recommendations . 
recomondations.foreach(new VoidFunction<Rating>() {  
  @Override  
  public void call(Rating rating) throws Exception {   
    String str = "User : " + rating.user() + //   " Product : " + rating.product() + //   " Rating : " + rating.rating();   
    System.out.println(str);  
    } 
  });
  
```


# Output
User : 2 Product : 3 Rating : 54.54927015541634

User : 1 Product : 4 Rating : 49.93948224984236

# Conclusion
The above output recommends the User 2 would like to buy Product 3 and  User 1 would go for User 4. 
This also recommends that their is no recommendation for new product as they do not match any similarity criteria in past.
