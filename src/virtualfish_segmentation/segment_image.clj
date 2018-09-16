(ns virtualfish-segmentation.segment-image
  (:require [fun.imagej.img :as img]
            [fun.imagej.img.cursor :as cursor]
            [fun.imagej.img.shape :as shape]
            [fun.imagej.img.type :as tpe]
            [fun.imagej.core :as ij]
            [fun.imagej.ops :as ops]
            [fun.imagej.img.utils :as img-utils]
            [fun.imagej.mesh :as msh]
            [fun.imagej.segmentation.fast-segmentation :as fseg]
            [brevis-utils.parameters :as params]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [virtualfish-segmentation.training-utils :as tutils]
            [loom.graph :as graph]
            [loom.alg :as graph-alg]))

(defn -main [& args]
    (let [;; First put everything into a map
          argmap (apply hash-map
                        (mapcat #(vector (read-string (first %)) (second %) #_(read-string (second %)))
                                (partition 2 args)))
          ;; Then read-string on *some* args, but ignore others
          argmap (apply hash-map
                        (apply concat
                               (for [[k v] argmap]
                                 [k (cond (= k :output-directory) v
                                          (= k :cache-directory) v
                                          (= k :directory) v
                                          (= k :basename) v
                                          (= k :rbc-filename) v
                                          (= k :registration-filename) v
                                          (= k :weight-filename) v
                                          (= k :output-filename) v
                                          :else (read-string v))])))
          arg-params (merge {:num-positive-samples 1000
                             :num-negative-samples 1000
                             :label-rbc true
                             :generate-dataset true
                             :solve-segmentation true
                             :save-segmentation-config true
                             :segmentation-config-filename "segmentation_config.clj"
                             :write-segmentation true
                             :segmentation-filename "segmentationMap.tif"
                             :basename "t00161_angle002_crop_001"
                             :directory "/Users/kharrington/Data/Daetwyler_Stephan/consecutiveData/t00161/"
                             :verbose true}
                             ; consider adding a tag
                             ; default params
                            argmap)
          basename (str (:directory arg-params)
                        (:basename arg-params))
          output-basename (str (:output-directory arg-params)
                               java.io.File/separator
                               (:basename arg-params))
          weight-filename (or (:weight-filename arg-params)
                              (str basename "_weights.csv"))]
      (doseq [[k v] arg-params]
        (params/set-param k v))
      (println args)
      (println argmap)
      (println arg-params)
      (println (:basename arg-params) (:directory arg-params) basename)

      (def full-image (fun.imagej.ops.convert/float32 (ij/open-img (str basename ".tif"))))
      ;(def vasculature (img/normalize (img/hyperslice full-image 2 0)))
      ;(def rbc (img/normalize (img/hyperslice full-image 2 1)))
      (def vasculature (img/normalize full-image))

      (def seg (atom (fseg/create-segmentation-meta (if (:load-segmentation-config arg-params)
                                                      (read-string (slurp (str basename (:segmentation-config-filename arg-params))))
                                                      {:num-positive-samples (:num-positive-samples arg-params)
                                                       :num-negative-samples (:num-negative-samples arg-params)
                                                       :basename (:basename arg-params)
                                                       :verbose (:verbose arg-params)
                                                       :cache-directory (if (:no-caching arg-params)
                                                                          nil
                                                                          (:cache-directory arg-params))
                                                       :segmentation-type :3D}))))
      
      (println "Adding feature maps")
      (reset! seg (tutils/setup-default-feature-maps @seg))

      (swap! seg assoc :weights (map read-string (string/split (slurp weight-filename) #",")))

      (println "Weights:")
      (println (:weights @seg))

      (when (:write-segmentation arg-params)
        (println "Writing segmentation map")
        (let [segmentation (fseg/segment-image @seg vasculature)]
         #_(println "writing Segmentation to\"" (str output-basename (params/get-param :segmentation-filename)) "\"" segmentation)
          #_(ij/save-img (fun.imagej.ops.copy/img segmentation)
                       (str output-basename (params/get-param :segmentation-filename)))
         #_(.save (.io ij/ij) segmentation
                                   (str "\"" basename (params/get-param :segmentation-filename) "\""))
          (let [background []#_(fun.imagej.ops.filter/gauss (fun.imagej.ops.create/img segmentation)
                                                        segmentation
                                                        (double-array [25 25 0]))
                to-threshold segmentation #_(fun.imagej.ops.math/subtract segmentation background)
                thresholded (fun.imagej.ops.morphology/dilate
                              (fun.imagej.ops.morphology/erode #_(fun.imagej.ops.threshold/otsu to-threshold)
                               (fun.imagej.ops.threshold/triangle to-threshold) ; li or triangle, used to be percentile
                               (shape/sphere 1))
                              (shape/sphere 1))]; was r 2
            (ij/save-img (fun.imagej.ops.convert/uint8 thresholded)
                         (str output-basename "_segmentation_thresholded.tif"))
           (let [mesh (msh/marching-cubes (fun.imagej.ops.convert/uint8 (ops/run-op "scaleView" (object-array [thresholded (double-array [0.25 0.25 1]) (net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory.)]))))]
                (println "Number of facets in mesh:" (.size (.getFacets mesh)))
                (msh/write-mesh-as-stl mesh (str output-basename ".stl")))
           #_(let [mesh (msh/marching-cubes thresholded)]
              (println "Number of facets in mesh:" (.size (.getFacets mesh)))
              (msh/write-mesh-as-stl mesh (str output-basename ".stl"))))))))

;(-main)
