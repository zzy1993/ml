import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by Aleph on 04/12/2016.
  */
object KCalculation {
  def isColumnNameLine(line:String):Boolean = {
    if (line != null && line.contains("Channel")) true
    else false
  }
  def main (args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KMeansClustering")
    val sc = new SparkContext(conf)

    val rawTrainingData = sc.textFile(args(0))
    val parsedTrainingData = rawTrainingData.filter(!isColumnNameLine(_)).map(line => {
      Vectors.dense(line.split(",").map(_.trim).filter(!"".equals(_)).map(_.toDouble))
    }).cache()
    val ks:Array[Int] = Array(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

    ks.foreach(cluster => {
      val model:KMeansModel = KMeans.train(parsedTrainingData, cluster,30,1)
      val ssd = model.computeCost(parsedTrainingData)
      println("sum of squared distances of points to their nearest center when k=" + cluster + " -> "+ ssd)
    })
  }
}
