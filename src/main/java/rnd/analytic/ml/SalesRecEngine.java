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

import scala.Tuple2;

@SuppressWarnings({ "serial", "rawtypes", "resource" })
public class SalesRecEngine {

	public static void main(String[] args) {

		// Turn off unnecessary logging
		java.util.logging.Logger.getGlobal().setLevel(java.util.logging.Level.OFF);
		org.apache.log4j.Logger.getRootLogger().setLevel(org.apache.log4j.Level.OFF);

		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("Sales Order Reccommandation Engine").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);

		int userIdIndex = 0;
		int userNameIndex = 1;
		int productIdIndex = 2;
		int productCodeIndex = 3;
		int rateIndex = 4;
		int qtyIndex = 5;
		int amountIndex = 6;
		
		// file format - user, product, rate, quantity, amount
		JavaRDD<String> salesOrdersFile = sc.textFile("target/classes/data/sales_orders.csv");

		int ratingIndex = amountIndex;
		// Map file to Ratings(user, item, rating) tuples
		JavaRDD<Rating> ratings = salesOrdersFile.map(new Function<String, Rating>() {
			public Rating call(String order) {
				String data[] = order.split(",");
				return new Rating(Integer.parseInt(data[userIdIndex]), Integer.parseInt(data[productIdIndex]), Double.parseDouble(data[ratingIndex]));
			}
		});

		// Build the recommendation model using ALS

		int rank = 10; // 10 latent factors
		int numIterations = 10; // number of iterations

		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations);

		// file format - user, product
		JavaRDD<String> salesLeadsFile = sc.textFile("target/classes/data/sales_leads.csv");

		// Create user-product tuples from leads
		JavaRDD<Tuple2<Object, Object>> userProducts = salesLeadsFile.map(new Function<String, Tuple2<Object, Object>>() {
			public Tuple2<Object, Object> call(String lead) {
				String data[] = lead.split(",");
				return new Tuple2<Object, Object>(Integer.parseInt(data[userIdIndex]), Integer.parseInt(data[productIdIndex]));
			}
		});

		// Predict the ratings of the products not rated by user
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

	private static Map createMap(JavaRDD<String> masterRDD) {

		final Map<Integer, String> map = new HashMap<Integer, String>();

		masterRDD.foreach(new VoidFunction<String>() {
			
			public void call(String row) {
				String data[] = row.split(",");
				map.put(Integer.parseInt(data[0]), data[0]);
			};
		});

		return map;
	}
}