from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim

spark = SparkSession.builder.appName("pyspark-movie-reviews").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = (
    spark.read.option("header", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv("IMDB Dataset.csv/IMDB Dataset.csv")
)

df.printSchema()
df.show(5, truncate=60)

df = df.filter(col("review").isNotNull() & (trim(col("review")) != ""))

print("Total reviews:", df.count())

print("Positive reviews:", df.filter(col("sentiment") == "positive").count())
print("Negative reviews:", df.filter(col("sentiment") == "negative").count())

df = df.withColumn("review_length", length(col("review")))

long_reviews = df.filter(col("review_length") > 500)
print("Reviews longer than 500 characters:", long_reviews.count())
long_reviews.show(10, truncate=60)

df.toPandas().to_csv("cleaned_movie_reviews.csv", index=False)

spark.stop()
