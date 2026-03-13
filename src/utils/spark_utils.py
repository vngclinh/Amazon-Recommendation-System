from pyspark.sql import SparkSession
from pyspark.sql.types import *


def get_spark_session(app_name="AmazonReviews", memory="24g"):
    """Khởi tạo SparkSession chuẩn — cả nhóm dùng hàm này."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", memory) \
        .config("spark.executor.memory", memory) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


def get_schema():
    """Schema chuẩn của dataset — cả nhóm dùng hàm này."""
    return StructType([
        StructField("marketplace",       StringType(),  True),
        StructField("customer_id",       StringType(),  True),
        StructField("review_id",         StringType(),  True),
        StructField("product_id",        StringType(),  True),
        StructField("product_parent",    StringType(),  True),
        StructField("product_title",     StringType(),  True),
        StructField("product_category",  StringType(),  True),
        StructField("star_rating",       IntegerType(), True),
        StructField("helpful_votes",     IntegerType(), True),
        StructField("total_votes",       IntegerType(), True),
        StructField("vine",              StringType(),  True),
        StructField("verified_purchase", StringType(),  True),
        StructField("review_headline",   StringType(),  True),
        StructField("review_body",       StringType(),  True),
        StructField("review_date",       DateType(),    True),
    ])


def load_data(spark, path, format="parquet"):
    """
    Load data dùng chung cho cả nhóm.

    Dùng:
        df = load_data(spark, "/kaggle/working/amazon_reviews_clean/")
        df = load_data(spark, "/kaggle/input/.../file.tsv", format="tsv")
    """
    if format == "parquet":
        return spark.read.parquet(path)
    elif format == "tsv":
        return spark.read \
            .option("sep", "\t") \
            .option("header", "true") \
            .option("quote", '"') \
            .option("escape", '"') \
            .option("multiLine", "true") \
            .option("mode", "PERMISSIVE") \
            .schema(get_schema()) \
            .csv(path)
    else:
        raise ValueError(f"Format không hợp lệ: {format}. Dùng 'parquet' hoặc 'tsv'.")
