# -- coding: gbk --
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
import random
import warnings
warnings.filterwarnings("ignore")

def build_vector(Matrix, coords, width=5, lower=1, positive=True, resolution=10000):
    window_size = 2 * width + 1
    
    i, j = np.indices((window_size, window_size))
    genomic_dist = resolution * np.sqrt(np.maximum((i-width)**2 + (j-width)**2, 0.1)) / 1000
    dist_matrix = np.maximum(genomic_dist, 10)**0.75
    
    def extract_ring_features(window):
        ring_features = []
        center = (width, width)
        
        # Define ring radius sequence (from center outward)
        radii = range(0, width+1)
        
        for r in radii:
            # Create ring mask
            mask = np.zeros_like(window, dtype=bool)
            
            # For center point
            if r == 0:
                mask[center] = True
            else:
                # Create points on the ring
                for angle in np.linspace(0, 2*np.pi, 8*r, endpoint=False):
                    x = int(round(center[0] + r * np.cos(angle)))
                    y = int(round(center[1] + r * np.sin(angle)))
                    
                    # Ensure within bounds
                    if 0 <= x < window_size and 0 <= y < window_size:
                        mask[x, y] = True
            
            # Calculate ring features
            ring_values = window[mask]
            if len(ring_values) > 0:
                ring_features.extend([
                    np.mean(ring_values),       # Mean value
                    np.max(ring_values),        # Maximum value
                    np.min(ring_values),        # Minimum value
                    np.median(ring_values),     # Median value
                    np.std(ring_values)         # Standard deviation
                ])
            else:
                ring_features.extend([0, 0, 0, 0, 0])
        
        return np.array(ring_features)
    
    for c in coords:
        x, y = int(c[0]), int(c[1])  # Ensure integer coordinates
        
        # Enhanced boundary checking
        valid_x = max(width, min(x, Matrix.shape[0]-width-1))
        valid_y = max(width, min(y, Matrix.shape[1]-width-1))
        if abs(valid_x - x) > 1 or abs(valid_y - y) > 1:
            continue  # Skip significantly offset coordinates
        
        try:
            # Safe window extraction
            x_start, x_end = valid_x-width, valid_x+width+1
            y_start, y_end = valid_y-width, valid_y+width+1
            
            window = Matrix[x_start:x_end, y_start:y_end].toarray()
            actual_size = window.shape[0] * window.shape[1]
            
            # Sparsity check
            if np.count_nonzero(window) < actual_size * 0.1:
                continue
                
            center_val = window[width, width]
            
            # ===== Feature extraction =====
            # Feature 1: Distance-corrected value features
            try:
                # Calculate only for valid windows
                corrected_window = window * dist_matrix[:window.shape[0], :window.shape[1]]
                dist_corrected_features = corrected_window.flatten()
            except:
                dist_corrected_features = np.zeros_like(window.flatten())
            
            # Feature 2: Ring features
            ring_features = extract_ring_features(window)
            
            # Feature 3: Center-to-background ratio
            try:
                # Optimized background region calculation: Use 10% edge area of window
                border_size = max(1, window.shape[0]//5)
                bg_rows = window[-border_size:, :]
                bg_cols = window[:, -border_size:]
                bg_region = np.concatenate((bg_rows.flatten(), bg_cols.flatten()))
                p2LL = center_val / (np.mean(bg_region) + 1e-7)
            except:
                p2LL = 0.0
            
            # Feature 4: Circular symmetry
            try:
                sym_matrix = np.abs(window - window.T)
                fro_window = np.linalg.norm(window, 'fro')
                fro_sym = np.linalg.norm(sym_matrix, 'fro')
                sym_score = 1 - fro_sym / (fro_window + 1e-7) if fro_window > 0 else 0
            except:
                sym_score = 0
            
            # Feature 5: Raw window features (preserve original spatial information)
            raw_window_features = window.flatten()
            
            # ===== Feature fusion =====
            features = np.hstack((
                dist_corrected_features,  # Feature 1: Distance-corrected values
                ring_features,             # Feature 2: Ring features
                raw_window_features,       # Feature 5: Raw window values
                np.array([p2LL]),         # Feature 3: Center-to-background ratio
                np.array([sym_score])     # Feature 4: Circular symmetry score
            ))
            
            # Feature vector length validation
            expected_size = (window_size**2 * 2) + (5 * (width + 1)) + 2
            if features.size == expected_size and np.all(np.isfinite(features)):
                yield features
                
        except Exception as e:
            # Print error message for debugging
            print(f"Error processing coordinates({x},{y}): {str(e)}")
            continue

def parsebed(chiafile, res=10000, lower=1, upper=5000000):
    # Parse BED file to extract genomic coordinates
    coords = defaultdict(set)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a //= res
            b //= res
            # Only include chromosome entries that don't contain 'M'
            if (b-a > lower) and (b-a < upper) and 'M' not in s[0]:
                chrom = 'chr' + s[0].lstrip('chr')
                coords[chrom].add((a, b))

    for c in coords:
        coords[c] = sorted(coords[c])

    return coords

def get_kde(coords):
    # Calculate Kernel Density Estimation for distance distribution
    dis = []
    for c in coords:
        for a, b in coords[c]:
            dis.append(b-a)

    lower = min(dis)

    kde = stats.gaussian_kde(dis)

    counts, bins = np.histogram(dis, bins=100)
    long_end = int(bins[-1])
    tp = np.where(np.diff(counts) >= 0)[0] + 2
    long_start = int(bins[tp[0]])

    return kde, lower, long_start, long_end

def negative_generating(M, kde, positives, lower, long_start, long_end):
    # Generate negative samples for training
    positives = set(positives)        
    N = 3 * len(positives)             
    # part 1: Sampling based on distance distribution
    part1 = kde.resample(N).astype(int).ravel()      
    part1 = part1[(part1 >= lower) & (part1 <= long_end)]

    # part 2: Uniform sampling for long-range interactions
    part2 = []
    pool = np.arange(long_start, long_end+1)   
    tmp = np.cumsum(M.shape[0]-pool)
    ref = tmp / tmp[-1]                        
    for i in range(N):                          
        r = np.random.random()                 
        ii = np.searchsorted(ref, r)            
        part2.append(pool[ii])                  

    sample_dis = Counter(list(part1) + part2)    

    neg_coords = []                        
    midx = np.arange(M.shape[0])          

    # Generate negative coordinates from distance pool
    for i in sorted(sample_dis):  
        n_d = sample_dis[i]                
        R, C = midx[:-i], midx[i:]           
        tmp = np.array(M[R, C]).ravel()   
        tmp[np.isnan(tmp)] = 0         
        mask = tmp > 0                 
        R, C = R[mask], C[mask]
        pool = set(zip(R, C)) - positives     
        sub = random.sample(pool, n_d)       
        neg_coords.extend(sub)     

    random.shuffle(neg_coords)    

    return neg_coords

