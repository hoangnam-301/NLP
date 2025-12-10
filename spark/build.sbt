ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "spark-nlp-labs",
    fork := true,
    // THÊM TẤT CẢ CÁC QUYỀN TRUY CẬP CẦN THIẾT CHO JAVA 17
    javaOptions ++= Seq(
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.nio.channels=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      // KHẮC PHỤC LỖI java.lang.invoke (Lỗi bạn đang gặp)
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      // KHẮC PHỤC LỖI java.util (Đã xảy ra trước đó)
      "--add-opens=java.base/java.util=ALL-UNNAMED" 
    ),
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.5.1",
      "org.apache.spark" %% "spark-sql" % "3.5.1",
      "org.apache.spark" %% "spark-mllib" % "3.5.1"
    )
  )