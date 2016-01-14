package ca.jakegreene.iris

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.Vectors

trait DataLoader {
  
  def irisFeatureColumn = "iris-features"
  def irisTypeColumn = "iris-type"
  /**
   * Load iris data.
   * 
   * The iris data is a collection of data collected by R.A. Fisher. It has measurements of various iris flowers
   * and is widely used for beginner statistics and machine-learning problems.
   * 
   * The data is a CSV with no header. It is in the format:
   * sepal length in cm, sepal width in cm, petal length in cm, petal width in cm, iris type
   * 
   * Example:
   * 5.1,3.5,1.4,0.2,Iris-setosa
   * 
   * @return a Dataframe with two columns. `irisFeatureColumn` contains the feature `Vector`s and `irisTypeColumn` contains the `String` iris types
   */
  def loadIris(filePath: String)(implicit sqlContext: SQLContext): DataFrame = {
    val irisData = sqlContext.sparkContext.textFile(filePath).flatMap { text =>
      text.split("\n").toList.map(_.split(",")).collect {
        case Array(sepalLength, sepalWidth, petalLength, petalWidth, irisType) =>
          (Vectors.dense(sepalLength.toDouble, sepalWidth.toDouble, petalLength.toDouble, petalWidth.toDouble), irisType)
      }
    }
    sqlContext.createDataFrame(irisData).toDF(irisFeatureColumn, irisTypeColumn)
  }
}