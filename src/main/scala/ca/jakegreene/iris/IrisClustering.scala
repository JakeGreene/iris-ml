package ca.jakegreene.iris

import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row

/**
 * IrisClustering is an experiment which uses clustering on iris data
 * and compares the clusters to the iris types.
 * 
 * Do the clusters align with the iris types?
 */
object IrisClustering extends DataLoader {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setAppName("iris-ml")
    val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    val irisDataFrame = loadIris("src/main/resources/iris.data")
    val (trainingData, testData) = {
      // Experiment with adjusting the size of the training set vs the test set
      val split = irisDataFrame.randomSplit(Array(0.8, 0.2))
      (split(0), split(1))
    }
    
    val cluster = new KMeans()
      .setK(3)
      .setFeaturesCol(irisFeatureColumn)
      
    val model = cluster.fit(trainingData)
    val predictions = model.transform(testData)
    
    predictions
      .select("prediction", irisTypeColumn)
      .collect()
      .foreach { case Row(prediction: Int, irisType: String) => 
        println(s"Assigned Cluster: $prediction \tIris Type: $irisType")
      }
  }  
}