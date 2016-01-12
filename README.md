Iris-ML
=======

A sample machine learning project using Apache Spark. 

Data
----

I am using R.A. Fisher's famous "iris" dataset, a dataset that contains 150 entries with 3 classifiers.

Usage
-----

This project is using spark 1.6.0 and scala 2.11
To run this project:

```
sbt clean assembly
${SPARK_1.6_HOME}/bin/spark-submit --class ca.jakegreene.iris.IrisClassification --master spark://127.0.0.1:7077 target/scala-2.11/iris.jar 
```
