Dynamic Outlier Aggregation with LOF on Trace Streams
===
This project contains the implementation of the *Dynamic Outlier Aggregation with LOF on Trace Streams*. Micro-clusters are dense deviations, which potentially hold significant information for the main process.
With this tool it is possible to detect such micro-clusters and classify them into own micro-cluster models. Synthetic event logs ([XES](http://www.xes-standard.org/openxes/start) show the applicability in the area of concept drift detection.

Usage
===
* To see first results, run the script with the prepared parameters. These are modified to cope with the synthetic logs.
* For further analysis of different real-world event logs, please have a look at the reference section, download the desired log and adjust PATH and parameter variables.
* It is also possible to import the main/reference model as a BPMN model. You can find the models corresponding to the synthetic log in the *bpmn* folder

References
===
* Besides the given synthetic logs, please feel encouraged to also apply the implementation on real-world logs which you can find in https://data.4tu.nl/repository/collection:event_logs_real. These have also been used to evaluate *DOA*
* The implemented framework is directly derived from the paper by Zellner, Richter, Sontheim, and Seidl.