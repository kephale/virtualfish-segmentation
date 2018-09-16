(ns virtualfish-segmentation.rbc-dataset
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
            [loom.graph :as graph]
            [loom.alg :as graph-alg]
            [virtualfish-segmentation.training-utils :as tutils]))

(defn generate-sample-points
  "Generate the positive and negative sample points given a segmentation and the target labeling.
      Positive samples are drawn from the labeling, while negative samples come from regions outside the labeling."
  [seg label]
  ;(println "generate-sample-points " (class label))
  (loop [seg (assoc seg
               :positive-samples #{}
               :negative-samples #{})]
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
          (and (or  (and (not (number? candidate-val)) (not candidate-val))
                    (and (number? candidate-val) (not (pos? candidate-val))))
               (< (count (:negative-samples seg)) (:num-negative-samples seg)))
          (do
            (when (:verbose seg)
              (println "pos:" (count (:positive-samples seg))
                       "neg:" (count (:negative-samples seg))))
            (recur (assoc seg
                     :negative-samples (conj (:negative-samples seg) candidate-pos))))
          :else
          (recur seg)))
      (assoc seg :positive-samples (vec (:positive-samples seg))
                 :negative-samples (vec (:negative-samples seg))))))

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
        arg-params (merge {:num-positive-samples 100
                           :num-negative-samples 100
                           :label-rbc true
                           :generate-dataset true
                           :solve-segmentation true
                           :save-segmentation-config true
                           :segmentation-config-filename "segmentation_config.clj"
                           :write-segmentation true
                           :segmentation-filename "segmentationMap.tif"
                           :basename "GFP_max"
                           :directory "/Volumes/rihla_store/Daetwyler_Stephan/segmented_vascular_data/171202_vasculatureMovie/tiff/e004/angle000/t00110/"
                           :output-directory "/Volumes/rihla_store/Daetwyler_Stephan/segmented_vascular_data/171202_vasculatureMovie/tiff/e004/angle000/t00110/output/"
                           :verbose true
                           :registration-filename "/Volumes/rihla_store/Daetwyler_Stephan/segmented_vascular_data/171202_vasculatureMovie/tiff/e004/angle000/t00110/registration_angle000.txt"}
                           ; consider adding a tag
                           ; default params
                          argmap)
        basename (str (:directory arg-params)
                      (:basename arg-params))
        output-basename (str (:output-directory arg-params)
                             java.io.File/separator
                             (:basename arg-params))
        weight-filename (str basename "_weights.csv")]
    (doseq [[k v] arg-params]
      (params/set-param k v))
    (println args)
    (println argmap)
    (println arg-params)
    (println (:basename arg-params) (:directory arg-params) basename)
    (def vasculature (fun.imagej.ops.convert/float32 (ij/open-img (str basename ".tif"))))
    (println "Using vasculature file: " (fun.imagej.ops.convert/float32 (ij/open-img (str basename ".tif"))) " " vasculature)
    (def rbc ^net.imglib2.RandomAccessibleInterval (img/create-img-like vasculature (net.imglib2.type.numeric.real.DoubleType.) #_(tpe/unsigned-byte-type)))


    (println "will write to: " (str (:output-directory arg-params) java.io.File/separator "RBC_mask.tif"))

    (when-not (.exists (java.io.File. (:output-directory arg-params)))
      (println "creating directory " (:output-directory arg-params))
      (.mkdirs (java.io.File. (:output-directory arg-params))))

    (let [registration-data (filter #(not (or (nil? %)
                                              (.isEmpty (.trim %))))
                                    (string/split-lines (slurp
                                                         (:registration-filename arg-params))))]
      (doseq [line registration-data]
        (println "regline " line)
        (let [parts (string/split line #":")
              tile-parts (string/split (first parts) #"_")
              tile-name (second tile-parts)
              reg-parts (string/split (second parts) #" ")
              xshift (read-string (nth reg-parts 2))
              yshift (read-string (nth reg-parts 4))
              position ^longs (long-array 3)
              _ (println (str (:directory arg-params) "RFP_" tile-name ".tif"))
              tile (ij/open-img (str (:directory arg-params) "RFP_" tile-name ".tif"))
              mmPair (fun.imagej.ops.stats/minMax tile)
              hmin (double (.get (.getA mmPair)))
              hmax (double (.get (.getB mmPair)))
              hdenom (- hmax hmin)
              ;_ (ij/show tile tile-name)
              ^net.imglib2.RandomAccess ra (.randomAccess rbc)
              movie-stack-length (/ (img/get-size-dimension tile 2) (img/get-size-dimension rbc 2))]
          (println "Thresholded")
          (img/map-img (fn [^net.imglib2.Cursor cur]
                         (.localize cur position)
                         (aset position 0 (long (+ (- (img/get-size-dimension tile 0) (aget position 0)) xshift))); x is flipped
                         (aset position 1 (long (+ (aget position 1) yshift)))
                         (aset position 2 (long (/ (aget position 2) movie-stack-length)));; Movie stack with 4 timesteps, we take only first frame
                         (when (and (not (neg? (aget position 0))) (< (aget position 0) (img/get-size-dimension rbc 0))
                                    (not (neg? (aget position 1))) (< (aget position 1) (img/get-size-dimension rbc 1))
                                    (not (neg? (aget position 2))) (< (aget position 2) (img/get-size-dimension rbc 2)))
                           (.setPosition ra position)
                           ;(println (seq position ) (cursor/get-val cur))
                           (.set ^net.imglib2.type.numeric.real.DoubleType (.get ra) ^double (max (/ (- (cursor/get-val cur) hmin) hdenom)
                                                                                                  (.get ^net.imglib2.type.numeric.real.DoubleType (.get ra))))

                           #_(.set ^net.imglib2.type.numeric.integer.UnsignedShortType (.get ra) ^int (max (/ (- (cursor/get-val cur) hmin) hdenom)
                                                                                                           (.get ^net.imglib2.type.numeric.integer.UnsignedShortType (.get ra)))); TODO: consider using max instead of +
                           #_(img/set-val rbc position ^boolean (cursor/get-val cur))))
                       tile)
          (println tile-name " " xshift " " yshift))))

    #_(def rbc-seg (fun.imagej.ops.math/multiply (fun.imagej.ops.threshold/moments rbc)
                                                 (fun.imagej.ops.threshold/moments vasculature)))
    (println "Starting backoff thresholding from moments:")
    (def rbc-seg
      (loop [rbc-thresh (fun.imagej.ops.threshold/moments (fun.imagej.ops.image/histogram rbc))]
        (println "Threshold level: " rbc-thresh)
        (let [rbc-seg (fun.imagej.ops.math/multiply (fun.imagej.ops.threshold/apply rbc rbc-thresh)
                                                    (fun.imagej.ops.threshold/moments vasculature))
              sum-count (.get (fun.imagej.ops.stats/sum rbc-seg))]
          (println "Sum count: " sum-count)
          (if (> sum-count
                 (:num-positive-samples arg-params))
            rbc-seg
            (recur (do (.dec rbc-thresh) rbc-thresh))))))
    (println "Writing: " rbc-seg " to " (str (:output-directory arg-params) java.io.File/separator "RBC_mask.tif"))


    (ij/save-img (fun.imagej.ops.convert/uint8 rbc-seg) (str (:output-directory arg-params) java.io.File/separator "RBC_mask.tif"));threshold used to be intermodes

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

    (when (:label-rbc arg-params)
      (println "Generating sample points.")
      (reset! seg (generate-sample-points @seg rbc-seg))
      (spit (str (:output-directory arg-params) java.io.File/separator "positive_samples.csv")
            (with-out-str
              (doseq [pt (:positive-samples @seg)]
                (println (string/join "," (seq pt))))))
      (spit (str (:output-directory arg-params) java.io.File/separator "negative_samples.csv")
            (with-out-str
              (doseq [pt (:negative-samples @seg)]
                (println (string/join "," (seq pt)))))))))    

;(-main)
