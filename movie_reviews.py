from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, explode, length, size, avg, count

spark = SparkSession.builder.appName("pyspark-movie-reviews").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = (
    spark.read.option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv("IMDB Dataset.csv/IMDB Dataset.csv")
)

print("Rows:", df.count())
df.printSchema()
df.show(5, truncate=60)

print("Sentiment counts")
df.groupBy("sentiment").count().show()

df = df.withColumn("clean", regexp_replace(col("review"), "<[^>]+>", " "))
df = df.withColumn("clean", lower(regexp_replace(col("clean"), "[^a-zA-Z\\s]", "")))
df = df.withColumn("words", split(col("clean"), "\\s+"))
df = df.withColumn("word_count", size(col("words")))

print("Average review length by sentiment")
df.groupBy("sentiment").agg(avg("word_count").alias("avg_words")).show()

stop_words = ["the", "a", "an", "and", "of", "to", "is", "in", "it", "this", "that", "i", "was", "as", "for", "with", "but", "on", "movie", "film", "are", "its", "be", "have", "his", "he", "you", "not", "they", "at", "one", "by", "from", "so", "all", "or", "who", "there", "has", "just", "about", "an", "were", "if", "out", "her", "she", "had", "what", "more", "my", "when", "very", "me", "can", "which", "their", "than", "no", "been", "will", "would", "do", "we", "some", "only", "into", "up", "even", "them", "other", "most", "also", "really", "much", "get", "dont"]

words = df.select("sentiment", explode(col("words")).alias("word"))
words = words.filter((length(col("word")) > 2) & (~col("word").isin(stop_words)))

for label in ["positive", "negative"]:
    print("Top 10 words in", label, "reviews")
    (
        words.filter(col("sentiment") == label)
        .groupBy("word")
        .agg(count("*").alias("total"))
        .orderBy(col("total").desc())
        .show(10)
    )

spark.stop()
