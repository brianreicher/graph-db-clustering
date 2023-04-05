ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "graph-nn"
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "org.apache.spark" %% "spark-streaming" % "3.2.0",
  "org.apache.kafka" % "kafka-clients" % "3.0.0",
  "org.apache.spark" %% "kafka" % "3.2.0",
)

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.4.8"
libraryDependencies += "org.apache.spark" %% "spark-streaming-kafka-0-8" % "2.4.8"
libraryDependencies += "org.twitter4j" % "twitter4j-stream" % "4.0.7"
