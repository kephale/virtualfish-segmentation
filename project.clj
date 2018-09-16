(defproject virtualfish-segmentation "0.1.0-SNAPSHOT"
  :description "Code to perform vascular segmentations described in 'Multi-sample SPIM image acquisition, processing and analysis of vascular growth in zebrafish' by Stephan Daetwyler, Carl Modes, Kyle Harrington, and Jan Huisken"
  :url "https://github.com/kephale/virtualfish-segmentation"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :jvm-opts ["-Xmx110g" "-server"]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [fun.imagej/fun.imagej "0.2.3-SNAPSHOT"]
                 [brevis.us/brevis-utils "0.1.1-SNAPSHOT"]
                 [seesaw "1.4.5"]
                 [aysylu/loom "1.0.0"]
                 [me.raynes/conch "0.8.0"]]
  :repositories [["imagej" "https://maven.imagej.net/content/groups/hosted/"]
                 ["imagej-releases" "https://maven.imagej.net/content/repositories/releases/"]
                 ["ome maven" "https://artifacts.openmicroscopy.org/artifactory/maven/"]
                 ["imagej-snapshots" "https://maven.imagej.net/content/repositories/snapshots/"]
                 ["sonatype-snapshots" "https://oss.sonatype.org/content/repositories/snapshots/"]
                 ["snapshots" {:url "https://clojars.org/repo"
                               :username :env/CI_DEPLOY_USERNAME
                               :password :env/CI_DEPLOY_PASSWORD
                               :sign-releases false}]
                 ["releases" {:url "https://clojars.org/repo"
                              :username :env/CI_DEPLOY_USERNAME
                              :password :env/CI_DEPLOY_PASSWORD
                              :sign-releases false}]])
