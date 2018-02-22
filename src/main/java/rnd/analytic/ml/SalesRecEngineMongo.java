package rnd.analytic.ml;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.bson.Document;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import com.mongodb.spark.rdd.api.java.JavaMongoRDD;

import scala.Tuple2;

@SuppressWarnings({ "serial", "resource" })
public class SalesRecEngineMongo {

	public static void main(String[] args) {

		// Turn off unnecessary logging
		java.util.logging.Logger.getGlobal().setLevel(java.util.logging.Level.OFF);
		org.apache.log4j.Logger.getRootLogger().setLevel(org.apache.log4j.Level.OFF);

		SparkConf conf = new SparkConf().//
				setAppName("rnd").//
				setMaster("local").//
				set("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/sparkdb.myCollection").//
				set("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/sparkdb.myCollection");

		JavaSparkContext jsc = new JavaSparkContext(conf);

		// Read sales order file. format - userCode, user, productCode, product, rate, quantity, amount
		JavaMongoRDD<Document> salesOrdersRDD = getJavaMongoRDD(jsc, "SalesOrders");

		// Map file to Ratings(user,item,rating) tuples
		JavaRDD<Rating> ratings = salesOrdersRDD.map(new Function<Document, Rating>() {
			public Rating call(Document d) {
				return new Rating(d.getInteger("userCode"), d.getInteger("productCode"), ((Number) d.get("amount")).doubleValue());
			}
		});

		// Build the recommendation model using ALS

		int rank = 10; // 10 latent factors
		int numIterations = 10; // number of iterations

		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations);

		// Read sales order file. format - userCode, user, productCode, product, rate, quantity, amount
		JavaMongoRDD<Document> salesLeadsRDD = getJavaMongoRDD(jsc, "SalesLeads");

		// Create user-item tuples from ratings
		JavaRDD<Tuple2<Object, Object>> userProducts = salesLeadsRDD.map(new Function<Document, Tuple2<Object, Object>>() {
			public Tuple2<Object, Object> call(Document d) {
				return new Tuple2<Object, Object>(d.getInteger("userCode"), d.getInteger("productCode"));
			}
		});

		// Predict the ratings of the items not rated by user for the user
		JavaRDD<Rating> recomondations = model.predict(userProducts.rdd()).toJavaRDD().distinct();

		// Sort the recommendations by rating in descending order
		recomondations = recomondations.sortBy(new Function<Rating, Double>() {
			@Override
			public Double call(Rating v1) throws Exception {
				return v1.rating();
			}

		}, false, 1);

		JavaRDD<Rating> topRecomondations = new JavaSparkContext(recomondations.context()).parallelize(recomondations.take(10));

		// Print the top recommendations for user 1.
		topRecomondations.foreach(new VoidFunction<Rating>() {
			@Override
			public void call(Rating rating) throws Exception {
				String str = "User : " + rating.user() + //
				" Product : " + rating.product() + //
				" Rating : " + rating.rating();
				System.out.println(str);
			}
		});

	}

	private static JavaMongoRDD<Document> getJavaMongoRDD(JavaSparkContext jsc, String collName) {

		Map<String, String> readOverrides = new HashMap<String, String>();
		readOverrides.put("collection", collName);
		readOverrides.put("readPreference.name", "secondaryPreferred");
		ReadConfig readConfig = ReadConfig.create(jsc).withOptions(readOverrides);

		JavaMongoRDD<Document> mongoRDD = MongoSpark.load(jsc, readConfig);
		return mongoRDD;
	}

}
