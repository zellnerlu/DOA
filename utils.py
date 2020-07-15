import itertools

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pm4py.algo.conformance.tokenreplay import factory as token_replay
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.objects.bpmn.importer import bpmn20 as bpmn_importer
from pm4py.objects.conversion.bpmn_to_petri import factory as bpmn_to_petri
from pm4py.objects.petri import networkx_graph
from scipy.cluster.hierarchy import dendrogram

DEP_THRESH = 0.8
FIT_THRESH = 1.0  # not mandatory
BOTTOM_LIM = 0.98
UPPER_LIM = 1.03

def convert_bpmn(path):
    bpmn_graph = bpmn_importer.import_bpmn(path)
    model, initial_marking, final_marking, elements_correspondence, inv_elements_correspondence, el_corr_keys_map = bpmn_to_petri.apply(
        bpmn_graph)
    return model, initial_marking, final_marking

def shuffle_log(log, sublog_size, switch=.05):
    gt_log = list(range(len(log)))
    temp_log = []
    gt_temp = []
    gt_result = {}
    for i in range(0, len(log), sublog_size):
        temp_log.append(log[i:i + sublog_size])
        gt_temp.append(gt_log[i:i + sublog_size])
    for i in range(len(gt_temp)):
        if i > 0:  # do not add part of log which resembles the reference model
            gt_result[i-1] = gt_temp[i]
        else:
            gt_result["rm"] = gt_temp[i]
    result = []
    slice_length = int(sublog_size*switch)
    for x in range(int(1/switch)):
        for y in range(len(temp_log)):
            result.append(temp_log[y][x:x + slice_length])  # append first x% from sublog to result
    return [item for sublist in result for item in sublist], gt_result

def perf_measure2(y_actuals, y_hats, border):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for k, vs in y_hats.items():
        for v in vs:
            found = False
            for i, js in y_actuals.items():
                if i is not "rm":
                    if v in js:
                        TP += 1  # if trace was found in a gt_mc slot
                        found = True
            if not found:
                FN += 1  # if trace was not found at all in gt
    gathering = list(y_hats.values())
    for i in y_actuals["rm"]:
        if i not in gathering:
            TN += 1
        else:
            FP += 1
    return TP, FP, FN, TN

def create_model(log, params={"dependency_thresh": DEP_THRESH}):
    model, initial_marking, final_marking = heuristics_miner.apply(log, parameters=params)
    #model, initial_marking, final_marking = alpha_miner.apply(log)
    #model, initial_marking, final_marking = inductive_miner.apply(log)
    return model, initial_marking, final_marking


def create_networkx_graph(net, initial_marking, final_marking):
    nx_graph, unique_source_corr, unique_sink_corr, inv_dict = \
        networkx_graph.create_networkx_undirected_graph(net, initial_marking, final_marking)
    inv_inv_dict = {y: x for x, y in inv_dict.items()}
    return nx_graph, inv_inv_dict

# apply token replay
def get_replay_result(log, model, initial_marking, final_marking):
    replay_result = token_replay.apply(log, model, initial_marking, final_marking)
    return replay_result

# symmetrize upper triangular matrix to a complete symmetric matrix
def symmetrize(a):
    if a.ndim == 2:
        return a + a.T - np.diag(a.diagonal())
    elif not a:  # matrix is empty
        print("Matrix is empty")
    else:
        print("Previous computation returns wrong matrix regarding dimensions")

def reduce_matrix(m):
    for i, r in enumerate(m):
        for j, val in enumerate(r):
            if j > i:
                m[i][j] = 0.0
    return m


def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = [c]
        prev_color = c
    segs.append(curr_seg)  # the final one
    return segs

def plot_gantt(colors, num, switch=None):
    vertical_bars = []
    if switch is not None:
        for i in range(num):
            if i % switch == 0:
                vertical_bars.append(i)
    bar_size = 5
    num_bars = len(set(colors))
    if "black" in set(colors):  # black is default color for gaps
        num_bars -= 1
    # Declaring a figure "gnt"
    fig, gnt = plt.subplots(figsize=(16, 9), dpi=800)
    # Setting Y-axis limits
    gnt.set_ylim(0, (num_bars+1)*15)  # number of colors plus top margin
    # Setting X-axis limits
    gnt.set_xlim(0, len(colors))
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Non-Conforming Trace Identifiers')
    gnt.set_ylabel('Micro-Clusters')
    # Setting ticks on y-axis
    yticks = [x+17.5 for x in range(num_bars*15) if x % 15 == 0]
    gnt.set_yticks(yticks)
    plt.xticks(fontsize=10)
    gnt.set_xticks(np.arange(len(colors))[0::int(len(colors)/10)])
    # Labelling ticks of y-axis
    #yticklabels = []
    #for x in reversed(range(num_bars)):
    #    if x == 0:
    #        yticklabels.append("$RM$")
    #    else:
    #        yticklabels.append("$MC_{"+str(x-1)+"}$")
    yticklabels = ["$MC_{"+str(x)+"}$" for x in reversed(range(num_bars))]
    gnt.set_yticklabels(yticklabels)
    segments = find_contiguous_colors(colors)
    barhs = {}
    for color in set(colors):
        if color != "black":
            barhs[color] = []
    start = 0
    for count, seg in enumerate(segments):
        end = start + len(seg)
        if seg[0] != 'black':
            barhs[seg[0]].append((start, len(seg)))
        start = end
    counter = 1
    barhs = {k: v for k, v in sorted(barhs.items(), key=lambda y: y[1][0][0], reverse=True)}  # sort by global tid
    print(barhs)
    for k, v in barhs.items():
        gnt.broken_barh(v, (counter*15, bar_size), facecolors=k)
        counter += 1

    for xc in vertical_bars:
        plt.axvline(x=xc, c="k")

    gnt.grid(True)
    black_line = mlines.Line2D([], [], color='black', label='Inter-Drift Distance')
    plt.legend(handles=[black_line])
    plt.savefig("results/gantt" + str(num) + ".png")

def plot_colored_subplots(tids, ys, colors, num):
    NUM_PLOTS = 5
    plt.figure(figsize=(2, 4), dpi=600)
    plt.ticklabel_format(useOffset=False)
    fig, axs = plt.subplots(NUM_PLOTS, 1)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    for i in range(NUM_PLOTS):
        segments = find_contiguous_colors(colors[i])
        plt.sca(axs[i])
        plt.xticks(np.arange(len(colors[i])), tids[i], fontsize=8, rotation=90)
        axs[i].plot(ys[i], lw=.75)
        start = 0
        for count, seg in enumerate(segments):
            end = start + len(seg)
            if seg[0] != 'black':
                axs[i].axvspan(start, end - 1, facecolor=seg[0], alpha=0.45, zorder=-100)
            start = end

    plt.savefig('results/LOF_comb_plot' + str(num) + '.png', edgecolor='black', facecolor='white', transparent=True)
    plt.clf()

def plot_multicolored_lines(x, y, colors, num, appendix, tids):
    segments = find_contiguous_colors(colors)
    plt.figure(figsize=(24, 6), dpi=125)

    ax = plt.gca()
    ax.plot(np.arange(len(y)), y, lw=.75)

    start = 0
    for count, seg in enumerate(segments):
        end = start + len(seg)
        if seg[0] != 'black':
            plt.axvspan(start, end-1, facecolor=seg[0], alpha=0.45, zorder=-100)
        start = end
    plt.gca().set_xticks(np.arange(len(colors)))
    plt.ticklabel_format(useOffset=False)
    ax.set_xticklabels(tids)
    plt.xticks(fontsize=14, rotation=90)
    plt.tight_layout()
    plt.ylabel('Local Outlier Factor')
    plt.xlabel('Non-Conforming Trace Identifiers')
    plt.savefig('results/LOF_abs_scores_' + str(num+1) + "_" + appendix + '_colored.png', edgecolor='black', facecolor='white', transparent=True)
    plt.clf()

def plot_LOF(y, num, appendix, tids):

    fig, axs = plt.subplots(2,)
    plt.ticklabel_format(useOffset=False)
    axs[0].plot(np.arange(len(y)), y, lw=.75)
    zipped = sorted(list(zip(y, tids)), key=lambda x: x[0])
    scores = [a for a, b in zipped]
    ids = [b for a, b in zipped]
    axs[1].plot(np.arange(len(scores)), scores, lw=.75)
    plt.sca(axs[0])
    plt.xticks(np.arange(len(y)), [], fontsize=8, rotation=90)
    plt.sca(axs[1])
    plt.xticks(np.arange(len(y)), [], fontsize=8, rotation=90)
    axs[0].set(ylabel='LOF')
    axs[1].set(xlabel='Non-Conforming Trace Identifiers', ylabel='LOF')
    plt.savefig('results/LOF_abs_scores_' + str(num + 1) + "_" + appendix + '.png', edgecolor='black',
                facecolor='white', transparent=True)
    plt.clf()


def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # print(linkage_matrix)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def separate_scores(scores, non_conforming_tid_offset, offset, replay_results_below_threshold):
    sim_dens_val = 1.0
    thresh = sim_dens_val + offset
    cluster = []
    outlier = []
    to_be_clustered = []
    count_below = 0
    count_above = 0
    for i, score in enumerate(scores):
        corresp_rrbt = replay_results_below_threshold[i]
        if not corresp_rrbt["already_clustered"] and score < thresh:
            to_be_clustered.append(i)
            cluster.append((i, score, corresp_rrbt["trace"], i + non_conforming_tid_offset))
            count_below += 1
        if score > thresh:
            outlier.append((i, score, corresp_rrbt["trace"], i + non_conforming_tid_offset))
            count_above += 1
    return cluster, outlier, to_be_clustered

def compare_trace_to_micro_cluster(trace, micro_cluster, tid):
    i = 0
    found = False
    for cluster in micro_cluster:
        micro_process_log, model, im, fm = cluster
        if not is_non_conforming(get_replay_result([trace], model, im, fm)[0], trace, tid):
            found = True
            break
        else:
            i += 1
    if found:
        micro_process_log, model, im, fm = micro_cluster[i]
        micro_process_log.append(trace)
        new_model, im, fm = create_model(micro_process_log)
        micro_cluster[i] = (micro_process_log, new_model, im, fm)
    return found, i



# filter traces based on fitness threshold
def is_non_conforming(trace_result, trace, tid):
    traces_below_threshold = []
    if not trace_result['trace_is_fit']:
    #if not trace_result['trace_is_fit'] and trace_result["trace_fitness"] < .8:
        d = trace_result.copy()
        d["trace"] = trace
        d["already_clustered"] = False
        d["tid"] = tid
        return d
    else:
        return None


def compute_distances(replay_results_below_threshold, current_trace, inv_dict, nx_graph):
    distances = []
    for rrbt in replay_results_below_threshold:
        if len(current_trace["transitions_with_problems"]) == 0 or len(rrbt["transitions_with_problems"]) == 0:
            distances.append(0.0)
        else:
            overall_hop_count = 0
            cart_prod_comb = list(itertools.product(rrbt["transitions_with_problems"], current_trace["transitions_with_problems"]))

            for source, target in cart_prod_comb:
                # search for key and return value which is a number in the networkx graph
                src_number = inv_dict.get(source)
                tar_number = inv_dict.get(target)
                if src_number is not None and tar_number is not None:  # does a penalty make sense here?
                    overall_hop_count += nx.shortest_path_length(nx_graph, src_number, tar_number)
            avg_hop_count = (overall_hop_count / len(cart_prod_comb))
            distances.append(avg_hop_count)
    return distances