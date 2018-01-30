# Machine Learning Starter

#

Starting with Machine Learning could be as complicate as hyped and as easy as Hello World if conquered with simple use case like Recommendation Engine ( RE ). 

The most popular choice for starting Machine Learning in java is Apache Spark, as it come with a special ML library/module with lots of simple to advance algorithm.

Recommendation is consider as Collaborative Filtering problem and Apache Spark has built-in algorithm to implement it.

#

As per definition by Apache Spark website
These techniques aim to fill in the missing entries of a user-item association matrix. spark.mllib currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. spark.mllib uses the alternating least squares (ALS) algorithm to learn these latent factors. The implementation in spark.mllib has the following parameters:

numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).

rank is the number of features to use (also referred to as the number of latent factors). 

iterations is the number of iterations of ALS to run. ALS typically converges to a reasonable solution in 20 iterations or less. 

lambda specifies the regularization parameter in ALS. 

implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data. 

alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations. 

#

Apache Spark mllib is available as maven dependency on central repository. You need to setup below module to get it started. 

```
<dependency> 
  <groupId>org.apache.spark</groupId>  
  <artifactId>spark-core_2.11</artifactId>  
  <version>${spark.version}</version> 
</dependency>

<dependency>  
  <groupId>org.apache.spark</groupId>  
  <artifactId>spark-mllib_2.11</artifactId>  
  <version>${spark.version}</version> 
</dependency>  

 ```

#

Now before getting your hand dirty with some code, you need to build valid data sets. 
In our case we are building a sample Sales Lead prediction model based on past Sales Orders. 
Here is few sample records from both data sets :


Sales Orders :


UserId UserName ProductId ProductName  Rate Quantity Amount

1  User 1  1   Product 1  10  5   50

1  User 1  2   Product 2  20  10   200

1  User 1  3   Product 3  10  15   150

2  User 2  1   Product 1  10  5   50

2  User 2  2   Product 2  20  20   400

2  User 2  4   Product 5  10  15   150

Sales Leads :


UserId UserName ProductId ProductName

1  User 1  4   Product 4

1  User 1  5   Product 5

2  User 2  3   Product 3

2  User 2  6   Product 6

We need to predict/recommend most relevant Product for both the user based onto their past order history. Here we can see Both User 1 and User 2 ordered Product 1 and Product 2, also they have ordered one item separately. 

Now we predicting their rating for alternate product and one new product.

#

First step is to load the training model and convert it to Rating format using JavaRDD API.

```
JavaRDD<String> salesOrdersFile = sc.textFile("target/classes/data/sales_orders.csv");

int ratingIndex = amountIndex; // Map file to Ratings(user, item, rating) tuples 

JavaRDD<Rating> ratings = salesOrdersFile.map(new Function<String, Rating>() {  
  public Rating call(String order) {   
    String data[] = order.split(",");   
    return new Rating(Integer.parseInt(data[userIdIndex]), Integer.parseInt(data[productIdIndex]), Double.parseDouble(data[ratingIndex]));  
    } 
  });
  
```

#

Next step is to train the Matrix Factorization model using ALS algorithm.

```
MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations); 
```

#

Now we load the Sales Lead file and convert it to Tupple format.

```
// file format - user, product 
JavaRDD<String> salesLeadsFile = sc.textFile("target/classes/data/sales_leads.csv");

// Create user-product tuples from leads
JavaRDD<Tuple2<Object, Object>> userProducts = salesLeadsFile.map(new Function<String, Tuple2<Object, Object>>() {  
  public Tuple2<Object, Object> call(String lead) {   
    String data[] = lead.split(",");   
    return new Tuple2<Object, Object>(Integer.parseInt(data[userIdIndex]), Integer.parseInt(data[productIdIndex]));  
  } 
});

```

#

Finally we can predict the future rating using simple API.

```
// Predict the ratings of the products not rated by user 
JavaRDD<Rating> recomondations = model.predict(userProducts.rdd()).toJavaRDD().distinct();

```

#

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

#

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
