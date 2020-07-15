
import logging
import sys
import time
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.visualization.petrinet import factory as pn_vis_factory

from sklearn.neighbors import LocalOutlierFactor
import os
from utils import *

#params
###
MAIN_LOG_SIZE = 1000  # number of traces in main log
SUB_LOG_SIZE = 1000  # number of traces in deviating log
SWITCH = .05  # defines inter-drift distance for synthetic log creation
# LOWER_BOUND defines the size at which the algorithm searches for LOF scores below threshold for the first time
# has to be lower or equal than SLIDING WINDOW SIZE
LOWER_BOUND = 45
SLIDING_WINDOW_SIZE = NUM_NEIGH = 50
OFFSET = 0.025  # how much is a LOF allowed to deviate from 1 to be counted as an outlier
DISCOVER = True
CREATE_PLOTS = True
PATH = os.path.join("datasets", "synthetic")
###

#logging
###
logging.basicConfig(filename='outlier_aggregration.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
root = logging.getLogger()
# Adding another logging handler for stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
###

def import_logs(path):
    l = []
    for i, file in enumerate(os.listdir(path)):
        logging.debug("Importing: {}".format(file))
        sublog = xes_import_factory.apply(os.path.join(path, file))
        logging.debug("Sublog length: {}".format(len(sublog)))

        for item in sublog:
            l.append(item)
    return l


log = import_logs(PATH)

logging.debug("Number of traces in whole log: {}".format(len(log)))
log_main = log[:MAIN_LOG_SIZE]
logging.debug("Number of traces in main log: {}".format(len(log_main)))

#del log[:MAIN_LOG_SIZE]  # remove main log, this is the log representing the trace stream
#logging.debug("Log size after deletion: {}".format(len(log)))

logging.debug("Creating process model...")

if DISCOVER:
    main_model, main_im, main_fm = create_model(log_main)
    #save_heuristics_net(log_main)
else:
    # reference model can also be imported instead of using process discovery
    main_model, main_im, main_fm = convert_bpmn(os.path.join("datasets", "bpmn_models", "1.bpmn"))
# visualize
gviz = pn_vis_factory.apply(main_model, main_im, main_fm, parameters={"format": "png"})
logging.debug("Saving micro-cluster model")
pn_vis_factory.save(gviz, "results/rm.png")

# introducing recurring drifts with specific inter-drift distance
log, gt = shuffle_log(log, SUB_LOG_SIZE, switch=SWITCH)
#logging.debug("Number of traces in whole log: {}".format(len(log)))

logging.debug("Saved micro-cluster model")

logging.debug("Creating networkx graph...")
nx_graph, inv_dict = create_networkx_graph(main_model, main_im, main_fm)
# only work on largest connected component
nx_graph = nx_graph.subgraph(max(nx.connected_components(nx_graph), key=len))
inv_dict = {list(inv_dict.keys())[list(inv_dict.values()).index(v)]: v for v in nx_graph.nodes}

logging.debug("Is connected: {}".format(nx.is_connected(nx_graph)))

LOCAL_LIM = len(log)

replay_results_below_threshold = []  # holds all rrbt; traces which are non-conforming
distances = []  # distance of a specific trace to the others
matrix = []  # holds all aforementioned distances (sometimes numpy array, sometimes list)

micro_cluster = []  # holds all micro-clusters (logs, models etc.)
non_conforming = 0  # counter for non-conforming traces in log
conforming = 0  # counter for conforming traces in log
mc_conforming = 0  # counter for conforming traces (to micro-clusters) in log
mc_conforming_dict = {}
mc_conforming_dict2 = {}

# helper vars
last_i = None
last_abscores = []
abs_scores = []
current_colors = []
global_colors = []

# number of colors should be >= expected number of micro-clusters for appropriate visualization
# blue is default color
palette = ["green", "yellow", "darkorchid",
           "red", "mediumspringgreen", "cyan",
           "turquoise", "slateblue", "deeppink",
           "lime", "darkorange", "blue",
           "bisque", "darkgreen", "blueviolet",
           "navajowhite", "dimgrey", "rosybrown",
           "tomato", "lightcyan", "aqua",
           "mediumpurple", "fuchsia", "olivedrab",
           "orangered", "sandybrown", "lightskyblue",
           "palegreen", "dodgerblue", "crimson",
           "indigo", "teal", "cadetblue",
           "peru", "gold", "violet"]
pointer = 0
gather_abs_scores = []
gather_current_colors = []
gather_tids = []
non_conforming_tid_offset = 0

global_start_time = time.time()
durations = []
sliding_window_full = False
for i, trace in enumerate(log):
    logging.debug("{}".format(i))
    global_colors.append("black")

    # tbt consists of replay_result of non-conforming trace with original trace and tid appended to dictionary
    # check if trace is non-conforming
    rrbt = is_non_conforming(get_replay_result([trace], main_model, main_im, main_fm)[0], trace, i)

    if rrbt:  # if trace is non-conforming to reference model
        logging.debug("Non-conforming trace")
        non_conforming += 1
        # check if non-conforming trace already fits to a micro-cluster

        trace_fits, found_index = compare_trace_to_micro_cluster(rrbt["trace"], micro_cluster, i)

        # if trace has not yet been assigned to a micro-cluster or the reference model
        if not trace_fits:
            comp_duration = 0
            if not isinstance(matrix, list):
                matrix = reduce_matrix(matrix).tolist()
            # only compute distances if there are traces to compare
            if len(replay_results_below_threshold) > 0:
                start_time = time.time()
                distances = compute_distances(replay_results_below_threshold, rrbt, inv_dict, nx_graph)
                comp_duration += time.time()-start_time

            # additionally append distance to itself
            distances.append(0.0)

            replay_results_below_threshold.append(rrbt)
            current_colors.append("black")

            tids = [rrbt["tid"] for rrbt in replay_results_below_threshold]

            start_time = time.time()
            # fill every row of matrix
            for k, row in enumerate(matrix):
                #logging.debug(i, row, len(distances), len(row))
                for x in range(len(distances) - len(row)):
                    row.append(0.0)
            matrix.append(distances)
            if len(replay_results_below_threshold) >= LOWER_BOUND:
                # prepare matrix for LOF computation
                matrix = symmetrize(np.array(matrix))

                #""" SLIDING WINDOW
                # sliding window
                if len(replay_results_below_threshold) > SLIDING_WINDOW_SIZE:
                    sliding_window_full = True
                    # delete first row and column from symmetric matrix
                    matrix = np.delete(np.delete(matrix, 0, 0), 0, 1)
                    del replay_results_below_threshold[0]
                    del tids[0]
                    del current_colors[0]
                    non_conforming_tid_offset += 1
                #"""

                # LOF computation
                clf = LocalOutlierFactor(n_neighbors=NUM_NEIGH, algorithm='brute', metric='precomputed')
                lof = clf.fit(matrix)
                old_abs_scores = abs_scores
                abs_scores = np.absolute(lof.negative_outlier_factor_)

                last_abscores = abs_scores
                last_i = i

                cluster, outlier, to_be_clustered = separate_scores(abs_scores, non_conforming_tid_offset, offset=OFFSET, replay_results_below_threshold=replay_results_below_threshold)
                comp_duration += time.time()-start_time
                if comp_duration > 0 and sliding_window_full:
                    durations.append(comp_duration)
                    sliding_window_full = False
                # Cluster traces if score is below threshold
                # Wait for at least NUM_NEIGH traces to form a micro-cluster
                if len(cluster) >= NUM_NEIGH:
                    logging.debug("Clustering at step/trace {} (including conforming and non-conforming steps)".format(i))

                    micro_process_log = []
                    for trace_index, _, cluster_trace, global_nc_id in cluster:
                        micro_process_log.append(cluster_trace)
                        current_colors[trace_index] = palette[pointer]
                        global_colors[i-trace_index] = palette[pointer]
                        if len(micro_cluster) not in mc_conforming_dict2:
                            mc_conforming_dict2[len(micro_cluster)] = []
                        mc_conforming_dict2[len(micro_cluster)].append(i-trace_index)

                    model, im, fm = create_model(micro_process_log)
                    micro_cluster.append((micro_process_log, model, im, fm))
                    mc_conforming_dict[len(micro_cluster)] = len(micro_process_log)

                    # Do not mark as clustered until it is really clustered
                    for ix in to_be_clustered:
                        replay_results_below_threshold[ix]["already_clustered"] = True

                    # visualize
                    logging.debug("Saving micro-cluster model #{}".format(len(micro_cluster)))
                    gviz = pn_vis_factory.apply(model, im, fm, parameters={"format": "png"})
                    pn_vis_factory.save(gviz, "results/mcm_" + str(i) + ".png")
                    logging.debug("Saved micro-cluster model")

                    if CREATE_PLOTS:
                        plot_range = np.arange(len(abs_scores))
                        plot_multicolored_lines(plot_range, abs_scores, current_colors, i, "unsorted", tids)
                        plot_LOF(abs_scores, i, "unsorted", tids)

                    gather_abs_scores.append(abs_scores)
                    gather_current_colors.append(current_colors.copy())
                    gather_tids.append(tids)

                    if CREATE_PLOTS:
                        plot_gantt(global_colors, i, SUB_LOG_SIZE*SWITCH)
                        #plot_gantt(global_colors, i)

                    logging.debug("Conforming: {}; MC-conforming: {}".format(conforming, mc_conforming))
                    if pointer < len(palette)-1:
                        pointer += 1  # jump to next color
                    else:
                        pointer = 0  # start again
                elif i % (SLIDING_WINDOW_SIZE*2) == 0 and CREATE_PLOTS:
                    plot_range = np.arange(len(abs_scores))
                    plot_multicolored_lines(plot_range, abs_scores, current_colors, i, "unsorted", tids)
                    plot_LOF(abs_scores, i, "unsorted", tids)
                    plot_gantt(global_colors, i, SUB_LOG_SIZE*SWITCH)
                    #plot_gantt(global_colors, i)
                    logging.debug("Conforming: {}; MC-conforming: {}".format(conforming, mc_conforming))
                if len(gather_abs_scores) == 5 and CREATE_PLOTS:
                    plot_colored_subplots(gather_tids, gather_abs_scores, gather_current_colors, i)
                    del gather_abs_scores[0]
                    del gather_current_colors[0]
                    del gather_tids[0]

        else:
            global_colors[len(global_colors)-1] = palette[found_index]
            logging.debug("Trace fits to micro-cluster: {}".format(found_index))
            mc_conforming += 1
            if found_index not in mc_conforming_dict:
                mc_conforming_dict[found_index] = 1
                mc_conforming_dict2[found_index] = []
                mc_conforming_dict2[found_index].append(i)
            else:
                mc_conforming_dict[found_index] += 1
                mc_conforming_dict2[found_index].append(i)
    else:
        logging.debug("Trace fits to main model")
        conforming += 1
    if i == LOCAL_LIM-1:
        plot_range = np.arange(len(abs_scores))
        plot_multicolored_lines(plot_range, last_abscores, current_colors, i, "unsorted", tids)
        plot_gantt(global_colors, i, SUB_LOG_SIZE*SWITCH)
        #plot_gantt(global_colors, i)

# Some concluding information
logging.debug("Took {}s".format(time.time()-global_start_time))
logging.debug("##### Number of micro-clusters: {}".format(len(micro_cluster)))
logging.debug("Conforming: {}; MC-conforming: {}".format(conforming, mc_conforming))
logging.debug("Number of traces conforming to MCMs: {}".format(mc_conforming_dict))
logging.debug("Number of traces conforming to MCMs: {}".format(mc_conforming_dict2))
logging.debug("Number of non-conforming traces: {}".format(non_conforming))
logging.debug("Max. duration: {}, Min. duration: {}, Avg.: {}".format(max(durations), min(durations), sum(durations)/len(durations)))

logging.debug("#######################")
tp, fp, fn, tn = perf_measure2(gt, mc_conforming_dict2, SWITCH*SUB_LOG_SIZE)
overall = tp+fp+fn+tn
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*((precision*recall)/(precision+recall))
accuracy = (tp + tn)/overall
logging.debug("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
logging.debug("Overall", overall)
logging.debug("Precision: {}".format(precision))
logging.debug("Recall: {}".format(recall))
logging.debug("F1-Score: {}".format(f1))
logging.debug("Accuracy: {}".format(accuracy))



