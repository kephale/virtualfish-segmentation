(ns virtualfish-segmentation.train-segmentation
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
            [virtualfish-segmentation.training-utils :as tutils]))

(defn generate-sample-points
  "Generate the positive and negative sample points given a segmentation and the target labeling.
      Positive samples are drawn from the labeling, while negative samples come from regions outside the labeling."
  [seg label]
  ;(println "generate-sample-points " (class label))
  (loop [seg (assoc seg
               :positive-samples []
               :negative-sampleso [])]
    (if (or (< (count (:positive-samples seg))
               (:num-positive-samples seg))
            (< (count (:negative-samples seg))
               (:num-negative-samples seg)))
      (let [candidate-pos (fseg/generate-position seg label)
            candidate-val (img/get-val ^net.imglib2.img.imageplus.ByteImagePlus label ^longs candidate-pos)]
        ;(println "Positive " (count (:positive-samples seg)) " Negative " (count (:negative-samples seg)) " val " candidate-val " position " (seq candidate-pos))
        (cond                                    ; True, and need positive samples
          (and (or  (and (not (number? candidate-val)) candidate-val)
                (and (number? candidate-val) (pos? candidate-val)))
               (< (count (:positive-samples seg)) (:num-positive-samples seg)))
          (do
            (when (:verbose seg)
              (println "pos:" (count (:positive-samples seg))
                       "neg:" (count (:negative-samples seg))))
            (recur (assoc seg
                     :positive-samples (conj (:positive-samples seg) candidate-pos))))
          ; False, and need negative samples
          (and (or (not candidate-val)
                   (zero? candidate-val))
               (< (count (:negative-samples seg)) (:num-negative-samples seg)))
          (do
            (when (:verbose seg)
              (println "pos:" (count (:positive-samples seg))
                       "neg:" (count (:negative-samples seg))))
            (recur (assoc seg
                     :negative-samples (conj (:negative-samples seg) candidate-pos))))
          :else
          (recur seg)))
      seg)))

(defn load-sample-points
  "Load the positive and negative sample points given a segmentation"
  [seg working-directory]
  (let [sample-sets (for [sample-filename ["positive_samples.csv" "negative_samples.csv"]]
                      (for [line (string/split-lines (slurp (str working-directory sample-filename)))]
                        (long-array (map #(Long/parseLong %) (string/split line #",")))))]
    (assoc seg
      :positive-samples (first sample-sets)
      :negative-samples (second sample-sets))))

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
                                        (= k :vasculature-filename) v
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
                           ;:cache-directory "/projects/VirtualFish/kyle/"
                           ;:cache-directory "/Users/kharrington/Data/Daetwyler_Stephan/test_ISVs/cache/"
                           :segmentation-config-filename "segmentation_config.clj"
                           :write-segmentation true
                           :segmentation-filename "segmentationMap.tif"
                           ;:basename "161122_angle001_t0188_registered"
                           :basename "GFP_max_crop"
                           :directory "/Volumes/rihla_store/Daetwyler_Stephan/segmented_vascular_data/171202_vasculatureMovie/tiff/e004/angle000/t00110/"
                           :output-directory "/Volumes/rihla_store/Daetwyler_Stephan/segmented_vascular_data/171202_vasculatureMovie/tiff/e004/angle000/t00110/output/"
                           :verbose true}
                           ; consider adding a tag
                           ; default params
                          argmap)
        basename (str (:directory arg-params)
                      (:basename arg-params))
        output-basename (str (:output-directory arg-params)
                             java.io.File/separator
                             (:basename arg-params))
        weight-filename (or (:weight-filename arg-params) (str output-basename "_weights.csv"))]
    (doseq [[k v] arg-params]
      (params/set-param k v))
    (println args)
    (println argmap)
    (println arg-params)
    (println (:basename arg-params) (:directory arg-params) basename)

;    (def full-image (fun.imagej.ops.convert/float32 (ij/open-img (str basename ".tif"))))
;    (def vasculature (img/normalize (img/hyperslice full-image 2 0)))
;    (def vasculature (img/normalize (fun.imagej.ops.convert/float32 (ij/open-img (str vasculature-filename ".tif")))))
;    (def rbc (img/normalize (fun.imagej.ops.convert/float32 (ij/open-img (str rbc-filename ".tif")))))

    (def vasculature (fun.imagej.ops.convert/float32 (ij/open-img (str basename ".tif"))))
;    (def rbc (img/create-img-like vasculature (tpe/bit-type)))

    (println "rbc-filename " (str (:rbc-filename arg-params)))
;     (def rbc (ij/open-img (str (:rbc-filename arg-params))))
    
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

    #_(when (:label-rbc arg-params)
        (println "Generating sample points.")
        (reset! seg (if (:rbc-presegmented arg-params)
                      (generate-sample-points @seg rbc #_(fun.imagej.ops.threshold/ij1 rbc))
                      (let [; try cleaning by taking median along z

                            histogram (fun.imagej.ops.image/histogram rbc)
                            thresh-val (first (fun.imagej.ops.threshold/intermodes histogram))
                            rbc-label (fun.imagej.ops.threshold/apply rbc thresh-val)]
                        (fseg/generate-sample-points @seg rbc-label)))))

    (when (:label-rbc arg-params)
        (println "Loading sample points.")
        (reset! seg (load-sample-points @seg (:output-directory arg-params))))

    (when (params/get-param :generate-dataset)
      (println "Generating dataset.")
      (reset! seg (fseg/generate-dataset @seg vasculature)))

    (when (:solve-segmentation arg-params)
      (println "Solving segmentation")
      (reset! seg (fseg/solve-segmentation @seg)))

    (println "Weights:")
    (println (:weights @seg))
    (spit weight-filename
          (string/join "," (:weights @seg)))

    (when (:save-segmentation-config arg-params)
      (println "Saving segmentation config")
      (fseg/save-segmentation-config @seg (:segmentation-config-filename arg-params)))))

;(-main)
