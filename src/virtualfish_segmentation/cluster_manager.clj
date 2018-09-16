(ns virtualfish-segmentation.cluster-manager
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
            [seesaw.core :as ss]
            [seesaw.mig :as mig]
            [seesaw.rsyntax :as rsyntax]
            [me.raynes.conch :refer [programs with-programs let-programs] :as sh])
  (:import [net.imglib2.img.array ArrayImgs]))

; Be warned that code is very customized to the MPI-CBG cluster, 
;   filesystem, and filenames/directories used in our paper

(defn get-default-parameter-filename
  []
  "/projects/VirtualFish/segmentation_parameters/global_001.csv")

(defn get-default-registration-filename
  []
  "/projects/VirtualFish/vascular_data/180112_vasculatureMovie/registration/registration_angle000.txt")

(defn get-default-target-vasculature
  []
  "GFP_max")

(defn get-default-per-parameter-filename
  []
  (str "parameter_" (System/nanoTime) ".csv"))

(defn get-default-fit-parameter-filename
  []
  (str "parameter_" (System/nanoTime) ".csv"))

(defn get-default-training-labels
  []
  (str "RBC_mask.tif"))

(def widget-set
  (let [experiment-listbox (ss/listbox :id :list :model [])]
    (atom {:title "Zebrafish Vasculature Segmentation"
           :launch-constant (ss/button :text "Launch constant")
           :launch-per-timepoint (ss/button :text "Launch per-timepoint")
           :launch-fit (ss/button :text "Launch fit")
           :hostname (ss/text :text "falcon")
           :username (ss/text :text "harringt")
           :experiment-listbox experiment-listbox
           :experiment-list (ss/scrollable experiment-listbox)
           :registration-filename (ss/text :text (get-default-registration-filename))
           :target-vasculature (ss/text :text (get-default-target-vasculature))
           :output-filename (ss/text :text (str (get-default-target-vasculature) "_segmentation"))
           :training-labels (ss/text :text (get-default-training-labels))
           :constant-parameter-filename (ss/text :text (get-default-parameter-filename))
           :per-timepoint-parameter-filename (ss/text :text (get-default-per-parameter-filename))
           :timepoint-range (ss/text :text "")
           :reload-experiments (ss/button :text "Reload experiments")
           :generate-dataset (ss/button :text "Generate dataset")})))

(defn get-output-directory
  ([]
   (get-output-directory (ss/selection (:experiment-listbox @widget-set))))
  ([path]
   (string/replace path
                   "vascular_data"
                   "segmented_vascular_data")))

; for autoupdating
(def experiment-list (ref []))

(defn list-model
  "Create list model based on collection"
  [items]
  (let [model (javax.swing.DefaultListModel.)]
    (doseq [item items] (.addElement model item))
    model))
; for autoupdating
(add-watch experiment-list nil
           (fn [_ _ _ items] (.setModel (:experiment-listbox @widget-set) (list-model items))))

(defn reload-experiments
  "Reload the list of experiments available to process on the cluster"
  [e]
  (with-programs [ssh]
                 (let [listing (string/split-lines
                                 (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)))
                                      "find /projects/VirtualFish/vascular_data/ -mindepth 4 -maxdepth 4 -type d -print -exec ls -ld  \\; | grep tiff"))]
                   (dosync
                     (ref-set experiment-list listing))
                   (println "ssh result:" listing))))

(defn primary-launcher
  "This is the primary launcher function, it does the real work."
  [argmap]
  (let [lein-location "/home/harringt/bin/lein"
        script-name (str "VirtualFish_" (System/nanoTime) ".slurm")
        code-directory "/projects/VirtualFish/kyle/timeseries_segmentation/zebrafish-vasculature"
        args (string/join " "
                          (map #(str %)
                              (interleave (keys argmap)
                                          (vals argmap))))
        dataset-command (str lein-location " run -m zebrafish-vasculature.rbc-dataset " args)
        train-command (str lein-location " run -m zebrafish-vasculature.train-segmentation " args)
        segment-command (str lein-location " run -m zebrafish-vasculature.segment-image " args)
        slurm-script
        (with-out-str (println "#!/bin/sh")
                      (println "#SBATCH -o \"/projects/VirtualFish/slurm_output/VirtualFish_%x_%J_%a.out\"")
                      (println)
                      (println "# sbatch --time=23:00:00 --mem=125000 " script-name)
                      (println)
                      (println "cd " code-directory ";")
                      (when (:generate-dataset argmap)
                        (println dataset-command))
                      (when (:fit-parameters argmap)
                        (println train-command))
                      (when (:segmentation argmap)
                        (println segment-command)))]
    (spit script-name slurm-script)
    (println "Slurm script")
    (println slurm-script)
    ; Submit slurm job
    script-name))

(defn initial-argmap
  []
  {:basename (ss/text (:target-vasculature @widget-set))
   :num-positive-samples 5000
   :num-negative-samples 5000
   :no-caching true
   :output-directory (get-output-directory)
   :registration-filename (ss/text (:registration-filename @widget-set))
   :verbose true})

(defn constant-launcher
  [e]
  (let [experiment-dir (ss/selection (:experiment-listbox @widget-set))
        base-argmap (assoc (initial-argmap)
                      :segmentation true
                      :weight-filename (ss/text (:constant-parameter-filename @widget-set)))]
                      
    (println "Experiment list" experiment-dir)
    (let [slurm-files
          (with-programs [ssh scp]
                         (let [listing (if (.isEmpty (ss/text (:timepoint-range @widget-set)))
                                         (string/split-lines
                                           (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)))
                                                "ls " experiment-dir))
                                         (let [chunks (string/split (ss/text (:timepoint-range @widget-set)) #";")
                                               expanded-chunks (for [chunk chunks]
                                                                 (if (.contains chunk "-")
                                                                   (let [startstop (string/split chunk #"-")]
                                                                     (range (read-string (first startstop)) (inc (read-string (last startstop)))))
                                                                   (read-string chunk)))
                                               times (flatten expanded-chunks)]
                                           (map #(format "t%05d" %) times)))
                               job-directory "/projects/VirtualFish/segmentation_jobs/"
                               slurm-files (doall
                                             (for [timepoint listing]
                                               (let [dirname (str experiment-dir java.io.File/separator timepoint java.io.File/separator)]
                                                 (primary-launcher (assoc base-argmap
                                                                     :output-filename (ss/text (:output-filename @widget-set))
                                                                     :output-directory (get-output-directory dirname)
                                                                     :directory dirname)))))]
                           (doseq [slurm slurm-files]
                             (scp slurm (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)) ":" job-directory))
                             (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set))) "sbatch --time=23:00:00 --mem=125000 " (str job-directory java.io.File/separator slurm)))))]
      (println slurm-files))))

(defn timepoint-launcher
  [e]
  (let [experiment-dir (ss/selection (:experiment-listbox @widget-set))
        base-argmap (assoc (initial-argmap)
                      :segmentation true)]
                      
    (println "Experiment list" experiment-dir)
    (let [slurm-files
          (with-programs [ssh scp]
                         (let [listing (if (.isEmpty (ss/text (:timepoint-range @widget-set)))
                                         (string/split-lines
                                           (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)))
                                                "ls " experiment-dir))
                                         (let [chunks (string/split (ss/text (:timepoint-range @widget-set)) #";")
                                               expanded-chunks (for [chunk chunks]
                                                                 (if (.contains chunk "-")
                                                                   (let [startstop (string/split chunk #"-")]
                                                                     (range (read-string (first startstop)) (inc (read-string (last startstop)))))
                                                                   (read-string chunk)))
                                               times (flatten expanded-chunks)]
                                           (map #(format "t%05d" %) times)))
                               job-directory "/projects/VirtualFish/segmentation_jobs/"
                               slurm-files (doall
                                             (for [timepoint listing]
                                               (let [dirname (str experiment-dir java.io.File/separator timepoint java.io.File/separator)]
                                                 (primary-launcher (assoc base-argmap
                                                                     :output-filename (ss/text (:output-filename @widget-set))
                                                                     :weight-filename (str (get-output-directory dirname) (ss/text (:per-timepoint-parameter-filename @widget-set)))
                                                                     :output-directory (get-output-directory dirname)
                                                                     :directory dirname)))))]
                           (doseq [slurm slurm-files]
                             (scp slurm (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)) ":" job-directory))
                             (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set))) "sbatch --time=23:00:00 --mem=125000 " (str job-directory java.io.File/separator slurm)))))]
      (println slurm-files))))

(defn fit-launcher
  [e]
  (let [experiment-dir (ss/selection (:experiment-listbox @widget-set))
        base-argmap (assoc (initial-argmap)
                      :fit-parameters true
                      :segmentation true)]
                      
    (println "Experiment list" experiment-dir)
    (let [slurm-files
          (with-programs [ssh scp]
                         (let [listing (if (.isEmpty (ss/text (:timepoint-range @widget-set)))
                                         (string/split-lines
                                           (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)))
                                                "ls " experiment-dir))
                                         (let [chunks (string/split (ss/text (:timepoint-range @widget-set)) #";")
                                               expanded-chunks (for [chunk chunks]
                                                                 (if (.contains chunk "-")
                                                                   (let [startstop (string/split chunk #"-")]
                                                                     (range (read-string (first startstop)) (inc (read-string (last startstop)))))
                                                                   (read-string chunk)))
                                               times (flatten expanded-chunks)]
                                           (map #(format "t%05d" %) times)))
                               job-directory "/projects/VirtualFish/segmentation_jobs/"
                               slurm-files (doall
                                             (for [timepoint listing]
                                               (let [dirname (str experiment-dir java.io.File/separator timepoint java.io.File/separator)]
                                                 (primary-launcher (assoc base-argmap
                                                                     :output-filename (ss/text (:output-filename @widget-set))
                                                                     :rbc-filename (str dirname (ss/text (:training-labels @widget-set)))
                                                                     :weight-filename (str (get-output-directory dirname) (ss/text (:per-timepoint-parameter-filename @widget-set)))
                                                                     :output-directory (get-output-directory dirname)
                                                                     :directory dirname)))))]
                           (doseq [slurm slurm-files]
                             (scp slurm (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)) ":" job-directory))
                             (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set))) "sbatch --time=23:00:00 --mem=125000 " (str job-directory java.io.File/separator slurm)))))]
      (println slurm-files))))

(defn generate-dataset
  [e]
  (let [experiment-dir (ss/selection (:experiment-listbox @widget-set))
        base-argmap (assoc (initial-argmap)
                      :generate-dataset true)]
                      
    (println "Experiment list" experiment-dir)
    (let [slurm-files
          (with-programs [ssh scp]
                         (let [listing (if (.isEmpty (ss/text (:timepoint-range @widget-set)))
                                         (string/split-lines
                                           (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)))
                                                "ls " experiment-dir))
                                         (let [chunks (string/split (ss/text (:timepoint-range @widget-set)) #";")
                                               expanded-chunks (for [chunk chunks]
                                                                 (if (.contains chunk "-")
                                                                   (let [startstop (string/split chunk #"-")]
                                                                     (range (read-string (first startstop)) (inc (read-string (last startstop)))))
                                                                   (read-string chunk)))
                                               times (flatten expanded-chunks)]
                                           (map #(format "t%05d" %) times)))
                               job-directory "/projects/VirtualFish/segmentation_jobs/"
                               slurm-files (doall
                                             (for [timepoint listing]
                                               (let [dirname (str experiment-dir java.io.File/separator timepoint java.io.File/separator)]
                                                 (primary-launcher (assoc base-argmap
                                                                     :output-filename (ss/text (:output-filename @widget-set))
                                                                     :rbc-filename (str dirname java.io.File/separator (ss/text (:training-labels @widget-set)))
                                                                     :output-directory (get-output-directory dirname)
                                                                     :directory dirname)))))]
                          (doseq [slurm slurm-files]
                              (scp slurm (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set)) ":" job-directory))
                              (ssh (str (ss/text (:username @widget-set)) "@" (ss/text (:hostname @widget-set))) "sbatch --time=23:00:00 --mem=125000 " (str job-directory java.io.File/separator slurm)))))]
      (println slurm-files))))

(defn -main
  [& args]
  ;; UI
  (let []
    (ss/listen (:launch-constant @widget-set)
               :action constant-launcher)
    (ss/listen (:launch-per-timepoint @widget-set)
               :action timepoint-launcher)
    (ss/listen (:launch-fit @widget-set)
               :action fit-launcher)
    (ss/listen (:reload-experiments @widget-set)
               :action reload-experiments)
    (ss/listen (:generate-dataset @widget-set)
               :action generate-dataset)
    (ss/invoke-later
      (-> (ss/frame :title (:title @widget-set),
                    :content (mig/mig-panel
                               :items [[(:title @widget-set) "span, gaptop 10"]
                                       [ :separator         "growx, wrap, gaptop 10"]
                                       [(ss/label :text "Cluster hostname") "gaptop 10"]
                                       [(:hostname @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Username") "gaptop 10"]
                                       [(:username @widget-set) "gaptop 10, growx, wrap"]
                                       [ :separator         "growx, wrap, gaptop 10"]
                                       [(ss/label :text "Experiment") "gaptop 10"]
                                       [(:experiment-list @widget-set) "growx, split, span, gaptop 10, wrap"]
                                       [(:reload-experiments @widget-set) "gaptop 10, wrap"]
                                       [ :separator         "growx, wrap, gaptop 10"]
                                       [(ss/label :text "Training label image") "gaptop 10"]
                                       [(:training-labels @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Target vasculature basename") "gaptop 10"]
                                       [(:target-vasculature @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Registration filename") "gaptop 10"]
                                       [(:registration-filename @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Constant Parameters") "gaptop 10"]
                                       [(:constant-parameter-filename @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Per-Timepoint Parameters") "gaptop 10"]
                                       [(:per-timepoint-parameter-filename @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Segmentation output filename") "gaptop 10"]
                                       [(:output-filename @widget-set) "gaptop 10, growx, wrap"]
                                       [(ss/label :text "Timepoint range (blank for everything, separate with semicolon, use dash for a range") "gaptop 10"]
                                       [(:timepoint-range @widget-set) "gaptop 10, growx, wrap"]
                                       [ :separator         "growx, wrap, gaptop 10"]
                                       [(:launch-constant @widget-set) "growx, gaptop 10, wrap"]
                                       [(:launch-per-timepoint @widget-set) "growx, gaptop 10, wrap"]
                                       [(:launch-fit @widget-set) "growx, wrap, gaptop 10"]
                                       [(:generate-dataset @widget-set) "growx, wrap, gaptop 10"]])
                    :on-close :exit)
          ss/pack!
          ss/show!))))

;(-main)
