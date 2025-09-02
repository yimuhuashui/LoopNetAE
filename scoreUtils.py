# -- coding: gbk --
import pathlib
import numpy as np
from scipy import sparse
from scipy import stats
import time
from tqdm import tqdm  

class Chromosome():
    def __init__(self, coomatrix, models, lower=1, upper=500, cname='chrm', res=10000, width=5):
        # Keep original initialization code unchanged
        R, C = coomatrix.nonzero()
        validmask = np.isfinite(coomatrix.data) & (C-R+1 > lower) & (C-R < upper)
        R, C, data = R[validmask], C[validmask], coomatrix.data[validmask]
        self.M = sparse.csr_matrix((data, (R, C)), shape=coomatrix.shape)
        self.ridx, self.cidx = R, C
        self.chromname = cname
        self.r = res
        self.w = width
        self.models = models
        
        # === New: Precompute distance matrix ===
        window_size = 2 * width + 1
        i, j = np.indices((window_size, window_size))
        genomic_dist = res * np.sqrt(np.maximum((i-width)**2 + (j-width)**2, 0.1)) / 1000
        self.dist_matrix = np.maximum(genomic_dist, 10)**0.75
    
    # === New: Ring feature extractor ===
    def extract_ring_features(self, window, width):
        """Extract ring features from center outward"""
        ring_features = []
        center = (width, width)
        window_size = 2 * width + 1
        
        for r in range(0, width+1):
            mask = np.zeros_like(window, dtype=bool)
            
            if r == 0:
                mask[center] = True
            else:
                for angle in np.linspace(0, 2*np.pi, 8*r, endpoint=False):
                    x = int(round(center[0] + r * np.cos(angle)))
                    y = int(round(center[1] + r * np.sin(angle)))
                    
                    if 0 <= x < window_size and 0 <= y < window_size:
                        mask[x, y] = True
            
            ring_values = window[mask]
            if len(ring_values) > 0:
                ring_features.extend([
                    np.mean(ring_values),
                    np.max(ring_values),
                    np.min(ring_values),
                    np.median(ring_values),
                    np.std(ring_values)
                ])
            else:
                ring_features.extend([0, 0, 0, 0, 0])
        
        return np.array(ring_features)
    
    def getwindow(self, coords):
        fts, clist = [], []
        width = self.w
        window_size = 2 * width + 1
        
        # Show progress with progress bar
        print(f"Processing {len(coords)} coordinates")
        for c in tqdm(coords, desc="Processing coordinates"):
            x, y = c[0], c[1]
            
            # Enhanced boundary check
            valid_x = max(width, min(x, self.M.shape[0]-width-1))
            valid_y = max(width, min(y, self.M.shape[1]-width-1))
            if abs(valid_x - x) > 1 or abs(valid_y - y) > 1:
                continue
            
            try:
                # Safe window extraction
                x_start, x_end = valid_x-width, valid_x+width+1
                y_start, y_end = valid_y-width, valid_y+width+1
                
                window = self.M[x_start:x_end, y_start:y_end].toarray()
                actual_size = window.shape[0] * window.shape[1]
                
                # Sparsity check
                if np.count_nonzero(window) < actual_size * 0.1:
                    continue
                
                center_val = window[width, width]
                
                # === Feature 1: Distance-corrected values ===
                try:
                    corrected_window = window * self.dist_matrix[:window.shape[0], :window.shape[1]]
                    dist_corrected_features = corrected_window.flatten()
                except:
                    dist_corrected_features = np.zeros_like(window.flatten())
                
                # === Feature 2: Ring features ===
                ring_features = self.extract_ring_features(window, width)
                
                # === Feature 3: Center to background ratio ===
                try:
                    border_size = max(1, window.shape[0]//5)
                    bg_rows = window[-border_size:, :]
                    bg_cols = window[:, -border_size:]
                    bg_region = np.concatenate((bg_rows.flatten(), bg_cols.flatten()))
                    p2LL = center_val / (np.mean(bg_region) + 1e-7)
                except:
                    p2LL = 0.0
                
                # === Feature 4: Ring symmetry ===
                try:
                    sym_matrix = np.abs(window - window.T)
                    fro_window = np.linalg.norm(window, 'fro')
                    fro_sym = np.linalg.norm(sym_matrix, 'fro')
                    sym_score = 1 - fro_sym / (fro_window + 1e-7) if fro_window > 0 else 0
                except:
                    sym_score = 0
                
                # === Feature 5: Raw window features ===
                raw_window_features = window.flatten()
                
                # === Feature fusion ===
                features = np.hstack((
                    dist_corrected_features,  # Feature 1
                    ring_features,             # Feature 2
                    raw_window_features,       # Feature 5
                    np.array([p2LL]),          # Feature 3
                    np.array([sym_score])      # Feature 4
                ))
                
                # Feature vector length validation
                expected_size = (window_size**2 * 2) + (5 * (width + 1)) + 2
                if features.size == expected_size and np.all(np.isfinite(features)):
                    fts.append(features.reshape(1, -1))
                    clist.append(c)
            except Exception as e:
                print(f"Error processing coordinate ({x},{y}): {str(e)}")
                continue
        
        if fts:
            test_x = np.vstack(fts)
        else:
            test_x = np.empty((0, 274))  # Feature dimension is 274 when width=5
        
        # Calculate actual feature dimension
        features = test_x.shape[1] if test_x.shape[0] > 0 else 274
        print(f"Extracted {len(test_x)} samples, feature dimension: {features}")
        
        # Reshape to 3D tensor (samples, features, 1)
        test_x_r = test_x.reshape((len(test_x), features, 1))
        print('Starting model prediction...')
        
        # Initialize probability array
        probas = np.zeros(len(test_x))
        
        # Iterate through each model
        for i, model in enumerate(tqdm(self.models, desc="Model prediction progress")):
            model_probs = model.predict(test_x_r, verbose=0, batch_size=4096)[:, 1]
            probas += model_probs
            print(f"Model {i+1}/{len(self.models)} prediction completed")
        
        probas /= len(self.models)
        
        print('Prediction completed.')
        return probas, clist

    def score(self, thre=0.5):
        # Keep original scoring code unchanged
        print('Scoring matrix {}'.format(self.chromname))
        print('Number of candidates {}'.format(self.M.data.size))
        coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
        print('---------Coordinate loading finished--------')
        p, clist = self.getwindow(coords)
        print('---------Windows obtained----------------')
        clist = np.r_[clist]
        pfilter = p > thre
        ri = clist[:, 0][pfilter]
        ci = clist[:, 1][pfilter]
        result = sparse.csr_matrix((p[pfilter], (ri, ci)), shape=self.M.shape)
        data = np.array(self.M[ri, ci]).ravel()
        self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

        return result, self.M

    def writeBed(self, out, prob_csr, raw_csr):
        # Keep original writing code unchanged
        print('---------Begin writing----------')
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
            r, c = prob_csr.nonzero()
            for i in range(r.size):
                line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
                        self.chromname, c[i]*self.r, (c[i]+1)*self.r,
                        prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]

                output_bed.write('\t'.join(list(map(str, line)))+'\n')