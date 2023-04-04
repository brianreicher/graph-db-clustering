import java.util.Properties
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.{Seconds, StreamingContext}

class Stream {
  def main(args: Array[String]): Unit = {

    // Set up Twitter API credentials
    val consumerKey = "YOUR_CONSUMER_KEY"
    val consumerSecret = "YOUR_CONSUMER_SECRET"
    val accessToken = "YOUR_ACCESS_TOKEN"
    val accessTokenSecret = "YOUR_ACCESS_TOKEN_SECRET"

    val filters = Seq("SEARCH_TERM") // Specify the search term to filter tweets

    // Set up the configuration for the Twitter stream
    val cb = new twitter4j.conf.ConfigurationBuilder()
      .setOAuthConsumerKey(consumerKey)
      .setOAuthConsumerSecret(consumerSecret)
      .setOAuthAccessToken(accessToken)
      .setOAuthAccessTokenSecret(accessTokenSecret)

    val auth = new twitter4j.auth.OAuthAuthorization(cb.build())
    val tweets = TwitterUtils.createStream(new StreamingContext(new SparkConf().setAppName("TwitterKafkaSparkStreaming"), Seconds(1)), Some(auth), filters)

    // Set up the Kafka producer properties
    val props = new Properties()
    props.put("bootstrap.servers", "YOUR_BOOTSTRAP_SERVERS")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

    // Create the Kafka producer
    val producer = new KafkaProducer[String, String](props)

    // Send the tweets to Kafka
    tweets.foreachRDD { rdd =>
      rdd.foreach { status =>
        val data = new ProducerRecord[String, String]("YOUR_TOPIC_NAME", status.getText)
        producer.send(data)
      }
    }

    // Set up the Spark Streaming context
    val ssc = new StreamingContext(new SparkConf().setAppName("TwitterKafkaSparkStreaming"), Seconds(1))

    // Set up the Kafka consumer properties
    val kafkaParams = Map[String, String](
      "bootstrap.servers" -> "YOUR_BOOTSTRAP_SERVERS",
      "key.deserializer" -> "org.apache.kafka.common.serialization.StringDeserializer",
      "value.deserializer" -> "org.apache.kafka.common.serialization.StringDeserializer",
      "group.id" -> "GROUP_ID"
    )

    // Create the Kafka stream
    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](Array("YOUR_TOPIC_NAME"), kafkaParams)
    )

    // Process the tweets and write them to a data log file
    val filteredTweets = stream.map(_.value()).filter(tweet => filterTweet(tweet)) // filter the tweets using the filterTweet function
    filteredTweets.foreachRDD { rdd =>
      rdd.coalesce(1).saveAsTextFile("YOUR_LOG_FILE_PATH") // write the tweets to a data log file
    }

    // Start the streaming context
    ssc.start()
    ssc.awaitTermination()
  }

  def filterTweet(tweet: String): Boolean = {
    // Implement your filtering logic here
    // This example function just returns true for all tweets
    true
  }
}

