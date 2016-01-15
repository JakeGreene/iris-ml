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
 * The primary question is: do the clusters align with the iris types?
 */
object IrisClustering extends DataLoader {
  /**
   * Only expects a single arg
   * arg(0) should have the path to the iris data
   */
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setAppName("iris-ml")
    val sc = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    val irisDataFrame = loadIris(args(0))
    val (trainingData, testData) = {
      // Experiment with adjusting the size of the training set vs the test set
      val split = irisDataFrame.randomSplit(Array(0.8, 0.2))
      (split(0), split(1))
    }
    
    /*
     *  No need for a pipeline here. KMeans will look at the feature vector and cluster (or "lump") the
     *  data into `K` groups (here we have chosen 3 groups) based on how similar their properties are.
     *  
     *  An important point: KMeans does not look at the labels i.e. it is an unsupervised learning
     *  algorithm. This is very useful if we have unlabeled data and we are looking to find related data
     *  e.g. given a document, find related documents.
     */
    val kmeans = new KMeans()
      .setK(3)
      .setFeaturesCol(irisFeatureColumn)
    
    // Create 3 clusters based on our training data
    val model = kmeans.fit(trainingData)
    
    /*
     *  For each flower in the test data, determine which cluster it should belong to.
     *  An iris will be assigned to a cluster based on which flowers in the training data
     *  it most closely resembles
     */
    val predictions = model.transform(testData)
    
    /*
     * Primary Question: Can KMeans accurately guess an iris' type without using the labels in training?
     * Hypothesis: Yes, the clusters created by KMeans will roughly align with the iris types
     *
     * This analysis makes the assumption that the majority of two or three iris types are not in a single cluster
     */
    val predsAndTypes = predictions.select("prediction", irisTypeColumn).collect().toList
    predsAndTypes.foreach { case Row(prediction: Int, irisType: String) =>
      println(s"Assigned Cluster: $prediction\tIris Type: $irisType")
    }
    
    val setosaAccuracy = accuracyOf("Iris-setosa", predsAndTypes)
    println(s"Accuracy of iris setosa is ${setosaAccuracy * 100}")
    val versicolorAccuracy = accuracyOf("Iris-versicolor", predsAndTypes)
    println(s"Accuracy of iris versicolor is ${versicolorAccuracy * 100}")
    val virginicasAccuracy = accuracyOf("Iris-virginica", predsAndTypes)
    println(s"Accuracy of iris virginicas is ${virginicasAccuracy * 100}")
    
    sc.stop()
  }
  
  /**
   * Determine how close the iris type `irisType` matches with its assigned cluster.
   */
  def accuracyOf(irisType: String, predsAndTypes: List[Row]): Double = {
    val clusters = predsAndTypes.collect {
      case Row(prediction: Int, iris: String) if iris == irisType => prediction
    }
    val cluster = mostCommon(clusters)
    clusters.filter(_ == cluster).size / clusters.size.toDouble
  }
  
  def mostCommon[A](l: List[A]): A = {
    l.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }
}