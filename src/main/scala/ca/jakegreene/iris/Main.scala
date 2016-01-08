package ca.jakegreene.iris

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object Main {
  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf(true).setAppName("iris-ml")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    /*
     * Data should be in the format:
     * Each line:
     * 1. sepal length in cm
     * 2. sepal width in cm
     * 3. petal length in cm
     * 4. petal width in cm
     * 5. class 
     * 
     */
    val irisData = sc.textFile("src/main/resources/iris.data").flatMap(_.split("\n").toList.map(_.split(",")).collect {
      case Array(sepalLength, sepalWidth, petaLength, petalWidth, irisType) =>
        (Vectors.dense(sepalLength.toDouble, sepalWidth.toDouble, petaLength.toDouble, petalWidth.toDouble), irisType)
    })
    // The ML pipeline requires that the data to be used is in a vector titled "features" and that the classifier is titled "label"
    val irisDataFrame = sqlContext.createDataFrame(irisData).toDF("features", "irisType")
    val (trainingData, testData) = {
      // Experiment with adjusting the size of the training set vs the test set
      val split = irisDataFrame.randomSplit(Array(0.8, 0.2))
      (split(0), split(1))
    }
    
    // Given String classes. These need to be indexed in order to work with the classifier
    val indexer = new StringIndexer()
      .setInputCol("irisType")
      .setOutputCol("label")
    val classifier = new RandomForestClassifier()
    val pipeline = new Pipeline()
      .setStages(Array(indexer, classifier))
      
    val model = pipeline.fit(trainingData)
    val predAndLabels = model.transform(testData)
        .select("prediction", "label")
        .map { case Row(prediction: Double, label: Double) => 
          (prediction, label)
        }
    
    // Determine how well we've done
    val metrics = new MulticlassMetrics(predAndLabels)
    println(s"Precision ${metrics.precision}")
    println(s"Recall ${metrics.recall}")
    println(s"F1 Score ${metrics.fMeasure}")
    sc.stop()
  }
}