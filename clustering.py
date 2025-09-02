# -- coding: gbk --
import gc
import numpy as np
from collections import defaultdict, Counter
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import euclidean
from scipy import sparse
import sys

def find_anchors(pos, min_count=3, min_dis=20000, wlen=800000, res=10000, merge_threshold=50000):
    """Find anchor regions on positions"""
    min_dis = max(min_dis // res, 1)
    wlen = min(wlen // res, 20)
    merge_threshold = merge_threshold // res  # Increase merge threshold

    count = Counter(pos)
    if not count:
        return set()
    
    min_idx, max_idx = min(count), max(count)
    refidx = range(min_idx, max_idx + 1)
    signal = np.array([count[i] for i in refidx])
    
    summits = find_peaks(signal, height=min_count, distance=min_dis)[0]
    sorted_summits = sorted([(signal[i], i) for i in summits], key=lambda x: -x[0])

    peaks = set()
    records = {}
    for _, i in sorted_summits:
        widths, height, left_ips, right_ips = peak_widths(
            signal, [i], rel_height=1, wlen=wlen
        )
        if len(left_ips) == 0 or len(right_ips) == 0:  
            continue
        
        li = int(np.round(left_ips[0]))
        ri = int(np.round(right_ips[0]))
        lb = refidx[li]
        rb = refidx[ri]
        current_summit = refidx[i]
        merged = False

        # Relax merge conditions
        for b in range(lb, rb + 1):
            if b in records:
                existing_summit, existing_lb, existing_rb = records[b]
                if abs(existing_summit - current_summit) > merge_threshold:
                    continue
                
                new_lb = min(lb, existing_lb)
                new_rb = max(rb, existing_rb)
                new_summit = existing_summit if signal[existing_summit-min_idx] >= signal[current_summit-min_idx] else current_summit
                
                peaks.remove((existing_summit, existing_lb, existing_rb))
                peaks.add((new_summit, new_lb, new_rb))
                
                for update_b in range(new_lb, new_rb + 1):
                    records[update_b] = (new_summit, new_lb, new_rb)
                merged = True
                break

        if not merged:
            peaks.add((current_summit, lb, rb))
            for b in range(lb, rb + 1):
                records[b] = (current_summit, lb, rb)

    return peaks

def _cluster_core(sort_list, r, visited, final_list):
    """Clustering core algorithm"""
    if not sort_list:
        return
    
    pos = np.array([coord for (_, coord) in sort_list])
    if len(pos) < 2:
        return
    
    # Use iterative clustering algorithm
    pool = set()
    for p in sort_list:
        score, coord = p
        coord_tuple = tuple(coord)
        if coord_tuple in pool:
            continue
        
        cen = coord
        rad = r
        Local = [coord_tuple]
        sub = [tuple(q) for q in pos if not np.array_equal(q, coord)]
        
        while sub:
            out = []
            for q in sub:
                if euclidean(q, cen) <= rad:
                    Local.append(q)
                else:
                    out.append(q)
            
            if len(out) == len(sub):  # No new points added
                break
                
            # Update center and radius
            cluster_arr = np.array(Local)
            cen = tuple(cluster_arr.mean(axis=0).round().astype(int))
            rad = max(euclidean(cen, q) for q in Local) + r
            sub = out
        
        # Save cluster
        final_list.append((coord_tuple, cen, int(rad)))
        pool.update(Local)
    
    visited.update(pool)

def local_clustering(Donuts, res, min_count=3, r=20000, max_keep=10):
    """Perform local clustering operation"""
    final_list = []
    if not Donuts:
        return final_list

    coords = list(Donuts.keys())
    x = np.array([c[0] for c in coords])
    y = np.array([c[1] for c in coords])

    x_anchors = find_anchors(x, min_count=min_count, min_dis=r, res=res)
    y_anchors = find_anchors(y, min_count=min_count, min_dis=r, res=res)
    r_cluster = max(r // res, 1)
    visited = set()

    # Anchor region clustering
    for x_a in x_anchors:
        for y_a in y_anchors:
            region_points = [
                (Donuts[(i, j)], (i, j)) 
                for i in range(x_a[1], x_a[2] + 1)
                for j in range(y_a[1], y_a[2] + 1)
                if (i, j) in Donuts
            ]
            if region_points:
                region_points.sort(reverse=True, key=lambda x: x[0])
                _cluster_core(region_points, r_cluster, visited, final_list)

    # Clustering outside anchor regions
    remaining_points = [
        (Donuts[coord], coord) 
        for coord in Donuts 
        if coord not in visited
    ]
    if remaining_points:
        remaining_points.sort(reverse=True, key=lambda x: x[0])
        _cluster_core(remaining_points, r_cluster, visited, final_list)

    # Summit point processing (relaxed requirements)
    x_summits = {a[0] for a in x_anchors}
    y_summits = {a[0] for a in y_anchors}
    for coord in Donuts:
        if coord in visited:
            continue
        if coord[0] in x_summits or coord[1] in y_summits:
            final_list.append((coord, coord, 0))

    return final_list  # Return all results directly

def process_chromosome(X, res, Athreshold):
    """Process chromosome data"""
    # Filter coordinates above threshold
    r = X[:, 0].astype(int) // res
    c = X[:, 1].astype(int) // res
    p = X[:, 2].astype(float)
    raw = X[:, 3].astype(float)
    
    # Initialize variables
    d = c - r
    tmpr, tmpc, tmpp, tmpraw, tmpd = r, c, p, raw, d
    matrix = {(r[i], c[i]): p[i] for i in range(len(r))}
    
    # Reduce coordinate count through iterative filtering
    count = 40001
    while count > 40000:
        D = defaultdict(float)
        P = defaultdict(float)
        unique_d = np.unique(tmpd)
        
        for distance in unique_d:
            dx = (tmpd == distance)
            dr, dc, dp, draw = tmpr[dx], tmpc[dx], tmpp[dx], tmpraw[dx]
            
            # Apply percentile filter
            pct_10 = np.percentile(dp, 10)
            dx = dp > pct_10
            dr, dc, dp, draw = dr[dx], dc[dx], dp[dx], draw[dx]
            
            # Accumulate values
            for i in range(dr.size):
                D[(dr[i], dc[i])] += draw[i]
                P[(dr[i], dc[i])] += dp[i]
        
        # Prepare for next iteration
        count = len(D)
        tmpr = np.array([k[0] for k in D])
        tmpc = np.array([k[1] for k in D])
        tmpp = np.array([P[k] for k in D])
        tmpraw = np.array([D[k] for k in D])
        tmpd = tmpc - tmpr
    
    return matrix, D

def write_output(Aoutfile, chrom, final_list, matrix, res):
    """Write filtered results to output file"""
    # Process coordinates
    r = [coord[0] for coord in final_list]
    c = [coord[1] for coord in final_list]
    p = np.array([matrix.get((r[i], c[i])) for i in range(len(r))])
    
    # Apply top-k filter if needed
    if len(r) > 7000:
        sorted_index = np.argsort(p)[-7000:]
        r = [r[i] for i in sorted_index]
        c = [c[i] for i in sorted_index]
    
    # Write to file
    with open(Aoutfile, 'w') as f:
        for i in range(len(r)):
            P_value = matrix.get((r[i], c[i]), 0)
            line = [chrom, r[i] * res, r[i] * res + res,
                    chrom, c[i] * res, c[i] * res + res, P_value]
            f.write('\t'.join(map(str, line)) + '\n')

def main(Ainfile, Aoutfile, Athreshold):
    """Main function to process input file and generate output"""
    # Configuration parameters
    Aresolution = 10000
    
    # Initialize dictionary to store chromosome data
    x = {}
    
    # Read input file
    try:
        with open(Ainfile, 'r') as source:
            for line in source:
                p = line.rstrip().split()
                if len(p) < 7:
                    continue
                    
                chrom = p[0]
                
                # Filter coordinates above threshold
                if float(p[6]) > Athreshold:
                    if chrom not in x:
                        x[chrom] = []
                    x[chrom].append([int(p[1]), int(p[4]), float(p[6]), float(p[7])])
    except Exception as e:
        print(f"File read error: {str(e)}")
        sys.exit(1)
    
    # Process each chromosome
    for chrom in x:
        # Convert to numpy array
        X = np.array(x[chrom])
        
        # Process chromosome data
        try:
            matrix, D = process_chromosome(X, Aresolution, Athreshold)
        except Exception as e:
            print(f"Error processing chromosome {chrom}: {str(e)}")
            continue
            
        # Free memory
        del X
        gc.collect()
        
        # Apply clustering
        try:
            final_list = [cluster[0] for cluster in local_clustering(D, res=Aresolution)]
            # Write results
            write_output(Aoutfile, chrom, final_list, matrix, Aresolution)
        except Exception as e:
            print(f"Clustering error for chromosome {chrom}: {str(e)}")
    
    if not x:
        print("Warning: No data found above threshold")

if __name__ == "__main__":
    # Move parameters here
    infile = "/home/yanghao/Loopnetae/candidate-loops/chr15.bed"
    outfile = "/home/yanghao/Loopnetae/loops/chr15.bedpe"
    threshold = 0.93
    
    # Call main function with parameters
    main(infile, outfile, threshold)