package ca.jakegreene.iris

import scala.reflect.runtime.universe
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.tuning.TrainValidationSplit

/**
 * IrisClassification is a simple example of Classification using Apache Spark's machine learning pipeline
 */
object IrisClassification {
  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf(true).setAppName("iris-ml")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    /*
     * iris.data is a collection of data collected by R.A. Fisher. It has measurements of various iris flowers
     * and is widely used for beginner statistics and machine-learning problems.
     * 
     * iris.data is a CSV file with no header. The data is in the format:
     * sepal length in cm, sepal width in cm, petal length in cm, petal width in cm, iris type
     * 
     * Example:
     * 5.1,3.5,1.4,0.2,Iris-setosa
     */
    val irisData = sc.textFile("src/main/resources/iris.data").flatMap { text =>
      text.split("\n").toList.map(_.split(",")).collect {
        case Array(sepalLength, sepalWidth, petaLength, petalWidth, irisType) =>
          (Vectors.dense(sepalLength.toDouble, sepalWidth.toDouble, petaLength.toDouble, petalWidth.toDouble), irisType)
      }
    }
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
     *  A multiclass classifier using a collection of decision trees. This classifier will create
     *  a model for predicting the "class" (i.e. iris type) of a flower based on its measurements
     *  
     *  Pipeline: Indexer -> Classifier
     */
    val indexer = new StringIndexer()
      .setInputCol(irisColumnName)
      .setOutputCol("label")
    // Classifiers look for the feature vectors under "features" and their labels under "label"
    val classifier = new RandomForestClassifier()
    val pipeline = new Pipeline()
      .setStages(Array(indexer, classifier))
      
    /*
     * There are a large number of "hyper" parameters that we can change to tune the accuracy
     * of our classifier. Instead of manually testing them, we can build a grid of parameters
     * and use a `TrainValidationSplit` to test the effectiveness of each combination
     */
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxDepth, Array(2, 5, 10))
      .addGrid(classifier.numTrees, Array(10, 20, 40))
      .addGrid(classifier.impurity, Array("gini", "entropy"))
      .build()
    
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      // Use 80% of the data to train and 20% to validate
      .setTrainRatio(0.8)
    
    // Create our model with the training-set of data
    val model = trainValidationSplit.fit(trainingData)
    
    // Use the model with our test-set of data
    val testResults = model.transform(testData)
    
    /*
     * Review the test. Our model has added the following columns:
     * - 'probability' a probability vector showing the odds the given flower is iris type iris_i for
     *    all i. e.g. [0.0, 0.4, 0.6] translates to 0% chance it is iris_0.0, 40% chance it
     *    is iris_1.0, 60% chance it is iris_2.0
     * - 'prediction' the label for the iris type that our model believes this row should be classified
     *    as. e.g. 2.0
     *    
     * We can compare the predicted label in `prediction` to the actual label in `label` to see how well
     * we did. A more advanced system might ignore predictions with a low probability in the `probability` vector
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