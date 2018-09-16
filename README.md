# virtualfish-segmentation

Github repository that accompanies the paper:

```
Multi-sample SPIM image acquisition, processing and analysis of vascular growth in zebrafish  
Stephan Daetwyler, Carl Modes, Kyle Harrington*, Jan Huisken*
*Co-corresponding authors
```

Contains code that is used to perform segmentations of zebrafish vasculature.

The code is customized for the MPI-CBG cluster, which as of this time, uses slurm as the job scheduler. Some paths and cluster-specific parameters are customized to the project but can be adjusted in the source code.

## Usage

The primary usage is via `virtualfish-segmentation.cluster-manager`. This namespace does not need to be run on the cluster, but will interact with the SLURM system on the target cluster.

The code was tested with SPIM images of whole zebrafish imaged using channels for: *Tg(kdrl:EGFP)* and *Tg(GATA1a:dsRed)*.

If you would like to run things locally, then:  

- `virtualfish-segmentation.rbc-dataset` - generates training data from both vasculature and RBC channels of a single timepoint  
- `virtualfish-segmentation.train-segmentation` - fit the segmentation parameters using training data, which could be generated with `virtualfish-segmentation.rbc-dataset`  
- `virtualfish-segmentation.segment-image` - generate a segmentation map for a given vasculature channel using a set of weights, which could be generated with `virtualfish-segmentation.train-segmentation`

## License

Copyright © 2018 Kyle Harrington

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
