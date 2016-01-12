Iris-ML
=======

A sample machine learning project using Apache Spark. 

Data
----

I am using R.A. Fisher's famous "iris" dataset, a dataset that contains 150 entries with 3 classifiers.

Usage
-----

This project is using Spark 1.6.0 and scala 2.11. Spark does not currently provide a 2.11 distribution, meaning you will need to spend ~15 minutes to [download](http://spark.apache.org/downloads.html) and [compile the source](http://spark.apache.org/docs/latest/building-spark.html#building-for-scala-211).

To use this project, run the following commands after setting or substituting SPARK_1.6_HOME to the spark 1.6.0 directory:

```
sbt clean assembly
${SPARK_1.6_HOME}/bin/spark-submit --class ca.jakegreene.iris.IrisClassification --master spark://127.0.0.1:7077 target/scala-2.11/iris.jar 
```
