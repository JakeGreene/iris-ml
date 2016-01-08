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
     * iris.data is a collection of data collected by R.A. Fisher. It has measurements of various iris flowers
     * and is widely used by beginner statistics and machine-learning problems
     * 
     * iris.data is a CSV file with no header in the format:
     * sepal length in cm, sepal width in cm, petal length in cm, petal width in cm, iris type
     * 
     * Example:
     * 5.1,3.5,1.4,0.2,Iris-setosa
     */
    val irisData = sc.textFile("src/main/resources/iris.data").flatMap(_.split("\n").toList.map(_.split(",")).collect {
      case Array(sepalLength, sepalWidth, petaLength, petalWidth, irisType) =>
        (Vectors.dense(sepalLength.toDouble, sepalWidth.toDouble, petaLength.toDouble, petalWidth.toDouble), irisType)
    })
    val irisColumnName = "iris-type"
    // The ML pipeline requires that the features used for learning are in a vector titled "features"
    val irisDataFrame = sqlContext.createDataFrame(irisData).toDF("features", irisColumnName)
    val (trainingData, testData) = {
      // Experiment with adjusting the size of the training set vs the test set
      val split = irisDataFrame.randomSplit(Array(0.8, 0.2))
      (split(0), split(1))
    }
    
    /*  Build the Pipeline
     *  
     *  StringIndexer:
     *  The iris types are all Strings. These need to be indexed (i.e. turned into unique doubles)
     *  in order to work with a classifier. e.g. "Iris-setosa" might become 1.0
     *  
     *  RandomForestClassifier:
     *  A multiclass classifier using a collection of decision trees.
     */
    val indexer = new StringIndexer()
      .setInputCol(irisColumnName)
      .setOutputCol("label")
    // Classifiers look for the feature vectors under "features" and their labels under "label"
    val classifier = new RandomForestClassifier()
    val pipeline = new Pipeline()
      .setStages(Array(indexer, classifier))
      
    // Create our model with the training-set of data
    val model = pipeline.fit(trainingData)
    
    // Test the model against our test-set of data
    val testResults = model.transform(testData)
    
    /*
     * Determine how well we've done. Our model has added the columns
     * - 'probability' a probability vector showing the odds the given row is for type iris_i for 
     *    all i. e.g. [0.0, 0.4, 0.6] (when translates to 0% iris_0.0, 40% iris_1.0, 60% iris_2.0)
     * - 'prediction' the label for the iris type that our model believes this row should be classified as. e.g. 2.0
     */
    val predAndLabels = testResults
        .select("prediction", "label")
        .map { case Row(prediction: Double, label: Double) => 
          (prediction, label)
        }
    val metrics = new MulticlassMetrics(predAndLabels)
    println(s"Precision ${metrics.precision}")
    println(s"Recall ${metrics.recall}")
    println(s"F1 Score ${metrics.fMeasure}")
    sc.stop()
  }
}