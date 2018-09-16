(ns virtualfish-segmentation.feature-generator
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

;; You can use this namespace to write feature maps to .tif

(defn -main [& args]
    (let [;; First put everything into a map
          argmap (apply hash-map
                        (mapcat #(vector (read-string (first %)) (second %))
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
                            argmap)
          basename (str (:directory arg-params)
                        (:basename arg-params))
          output-basename (str (:output-directory arg-params)
                               java.io.File/separator
                               (:basename arg-params))]
      (doseq [[k v] arg-params]
        (params/set-param k v))
      (println args)
      (println argmap)
      (println arg-params)
      (println (:basename arg-params) (:directory arg-params) basename)

      (def full-image (fun.imagej.ops.convert/float32 (ij/open-img (str basename ".tif"))))
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

      (doseq [feature-map-fn (:feature-map-fns @seg)]
        (let [feature-map-filename (str output-basename "_" (:name feature-map-fn) ".tif")
              feature-map ((:fn feature-map-fn) vasculature)]
            (ij/save-img feature-map feature-map-filename)))))

;(-main)
