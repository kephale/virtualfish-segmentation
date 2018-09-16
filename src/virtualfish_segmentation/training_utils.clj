(ns virtualfish-segmentation.training-utils
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
            [clojure.string :as string]))

(defn gradient-from-target
  "Return the hessians from a target"
  [target d]
  (img/gradient target d))
(def gradient-from-target-memo (memoize gradient-from-target))

(defn hessians-from-target
  "Return the hessians from a target"
  [target]
  (let [grads (img/concat-imgs [(img/gradient target 0) (img/gradient target 1) (img/gradient target 2)])]
    (img/hessian-matrix grads)))
(def hessians-from-target-memo (memoize hessians-from-target))

(defn eigens-from-hessian
  "Return the eigens from a hessian"
  [hessian]
  (img/tensor-eigen-values hessian))
(def eigens-from-hessian-memo (memoize eigens-from-hessian))

(defn write-vertices-to-xyz
  "Write a list of vertices to xyz."
  [verts filename]
  (spit filename
        (with-out-str
          (doall
            (for [vert verts]
              (println (string/join "\t" (seq vert))))))))

(defn generate-sample-points-negative-label
  "Generate the positive and negative sample points given a segmentation and the target labeling.
      Positive samples are drawn from the labeling, while negative samples come from regions outside the labeling."
  [seg label negative-label]
  (loop [seg seg]
    (if (or (< (count (:positive-samples seg))
               (:num-positive-samples seg))
            (< (count (:negative-samples seg))
               (:num-negative-samples seg)))
      (let [candidate-pos (fseg/generate-position seg label)
            candidate-val (img/get-val ^net.imglib2.img.imageplus.ByteImagePlus label ^longs candidate-pos)
            negative-candidate-val (img/get-val ^net.imglib2.img.imageplus.ByteImagePlus negative-label ^longs candidate-pos)]
        ;(println "Positive " (count (:positive-samples seg)) " Negative " (count (:negative-samples seg)) " val " candidate-val)
        (cond                                    ; True, and need positive samples
          (and candidate-val
               (< (count (:positive-samples seg)) (:num-positive-samples seg)))
          (do
            (when (:verbose seg)
              (println "pos:" (count (:positive-samples seg))
                       "neg:" (count (:negative-samples seg))))
            (recur (assoc seg
                     :positive-samples (conj (:positive-samples seg) candidate-pos))))
          ; False, and need negative samples
          (and negative-candidate-val
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

(defn setup-default-feature-maps
  "Setup the default feature maps that are used in mutliple namespaces"
  [seg]
  (-> seg
      (fseg/add-feature-map-fn "raw" (fn [target] (fun.imagej.ops.convert/float32 (img/copy target)))) ; Raw image as feature map
      #_(fseg/add-feature-map-fn "isoDataThresh_2.5" (fn [target]
                                                      (let [input (img/copy target)
                                                            hist (fun.imagej.ops.image/histogram input)
                                                            thresh-val (* 2.5 (.get (first (fun.imagej.ops.threshold/intermodes hist))))]
                                                        (fun.imagej.ops.convert/float32 (fun.imagej.ops.threshold/apply input (net.imglib2.type.numeric.real.FloatType. thresh-val))))))
      #_(fseg/add-feature-map-fn "li" (fn [target]
                                        (let [input (img/copy target)]
                                          (fun.imagej.ops.convert/float32 (fun.imagej.ops.threshold/li input)))))
      #_(fseg/add-feature-map-fn "varianceFilter3sphere_huang" (fn [target] (let [output (img/create-img-like target (tpe/double-type))]
                                                                             (fun.imagej.ops.filter/variance output
                                                                                                             target
                                                                                                             (shape/sphere-shape 3))
                                                                             (fun.imagej.ops.convert/float32 (fun.imagej.ops.threshold/huang output)))))
      (fseg/add-feature-map-fn "sqgradient_x" (fn [target]
                                                (let [grad (gradient-from-target-memo target 0)]
                                                  (fun.imagej.ops.math/multiply grad grad))))
      (fseg/add-feature-map-fn "sqgradient_y" (fn [target]
                                                (let [grad (gradient-from-target-memo target 1)]
                                                  (fun.imagej.ops.math/multiply grad grad))))
      (fseg/add-feature-map-fn "sqgradient_z" (fn [target]
                                                (let [grad (gradient-from-target-memo target 2)]
                                                  (fun.imagej.ops.math/multiply grad grad))))
      (fseg/add-feature-map-fn "sqgradient_all" (fn [target]
                                                  (let [xgrad (gradient-from-target-memo target 0)
                                                        ygrad (gradient-from-target-memo target 1)
                                                        zgrad (gradient-from-target-memo target 2)]
                                                    (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply zgrad zgrad)
                                                                             (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply xgrad xgrad)
                                                                                                      (fun.imagej.ops.math/multiply ygrad ygrad))))))
      #_(fseg/add-feature-map-fn "smooth_sqgradient_all" (fn [target]
                                                           (let [xgrad (img/gradient target 0)
                                                                 ygrad (img/gradient target 1)
                                                                 zgrad (img/gradient target 2)]
                                                             (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply zgrad zgrad)
                                                                                      (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply xgrad xgrad)
                                                                                                               (fun.imagej.ops.math/multiply ygrad ygrad))))))
      #_(fseg/add-feature-map-fn "smooth_sqgradient_all_by_raw" (fn [target]
                                                                  (let [xgrad (img/gradient target 0)
                                                                        ygrad (img/gradient target 1)
                                                                        zgrad (img/gradient target 2)]
                                                                    (fun.imagej.ops.math/multiply (img/copy target)
                                                                                                  (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply zgrad zgrad)
                                                                                                                           (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply xgrad xgrad)
                                                                                                                                                    (fun.imagej.ops.math/multiply ygrad ygrad)))))))
      (fseg/add-feature-map-fn "invgradient_by_input" (fn [target]
                                                        (let [xgrad (gradient-from-target-memo target 0)
                                                              ygrad (gradient-from-target-memo target 1)
                                                              zgrad (gradient-from-target-memo target 2)]
                                                          (fun.imagej.ops.math/multiply (img/copy target)
                                                                                        (fun.imagej.ops.image/invert (fun.imagej.ops.create/img target)
                                                                                                                     (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply zgrad zgrad)
                                                                                                                                              (fun.imagej.ops.math/add (fun.imagej.ops.math/multiply xgrad xgrad)
                                                                                                                                                                       (fun.imagej.ops.math/multiply ygrad ygrad))))))))
      #_(fseg/add-feature-map-fn "dog_3_2" (fn [target] (let [output (fun.imagej.ops.convert/float32 (img/create-img-like target))]
                                                          (fun.imagej.ops.filter/dog output
                                                                                     (fun.imagej.ops.convert/float32 (img/copy target))
                                                                                     3.0 2.0)
                                                          output)))
      #_(fseg/add-feature-map-fn "varianceFilter3sphere_dog_3_2" (fn [target] (let [output (img/create-img-like target (tpe/double-type))
                                                                                    dog-output (fun.imagej.ops.convert/float32 (img/create-img-like target))]
                                                                                (fun.imagej.ops.filter/variance output
                                                                                                                target
                                                                                                                (shape/sphere-shape 3))
                                                                                (fun.imagej.ops.filter/dog dog-output
                                                                                                           (fun.imagej.ops.convert/float32 (img/copy output))
                                                                                                           3.0 2.0))))
      #_(fseg/add-feature-map-fn "hessian_0" (fn [target]
                                              (img/hyperslice (hessians-from-target-memo target) 3 0)))
      #_(fseg/add-feature-map-fn "hessian_1" (fn [target]
                                              (img/hyperslice (hessians-from-target-memo target) 3 1)))
      #_(fseg/add-feature-map-fn "hessian_2" (fn [target]
                                              (img/hyperslice (hessians-from-target-memo target) 3 2)))
      #_(fseg/add-feature-map-fn "hessian_3" (fn [target]
                                              (img/hyperslice (hessians-from-target-memo target) 3 3)))
      #_(fseg/add-feature-map-fn "eigen_0" (fn [target]
                                            (img/hyperslice (eigens-from-hessian-memo (hessians-from-target-memo target)) 3 0)))
      #_(fseg/add-feature-map-fn "eigen_1" (fn [target]
                                            (img/hyperslice (eigens-from-hessian-memo (hessians-from-target-memo target)) 3 1)))))
