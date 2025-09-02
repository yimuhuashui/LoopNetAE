# -- coding: gbk --
import pathlib
import numpy as np
import trainUtils, utils
import warnings
from collections import defaultdict
import joblib
import loopnete
import random
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train(X, F, chromname, learning_rate, epochs, kernel_size, Aoutput):  
    """Training function - processes training data only"""
    # Combine positive and negative training samples
    all_x = np.vstack((X, F))
    all_y = np.hstack((np.ones(X.shape[0]), np.zeros(F.shape[0])))
    
    # Print class distribution
    unique, counts = np.unique(all_y, return_counts=True)
    print(f"{chromname} class distribution: Positive samples {counts[1]}, Negative samples {counts[0]}, Ratio 1:{counts[0]/counts[1]:.2f}")
    
    # Split training and validation sets (for training monitoring)
    train_x, val_x, train_y, val_y = train_test_split(all_x, all_y, test_size=0.1, stratify=all_y, random_state=42)
    
    # Convert labels to categorical format
    train_labels = to_categorical(train_y, num_classes=2)
    val_labels = to_categorical(val_y, num_classes=2)
    
    # Reshape for CNN input
    features = 274
    train_x_3d = train_x.reshape(train_x.shape[0], features, 1)
    val_x_3d = val_x.reshape(val_x.shape[0], features, 1)
    
    # Create and train model (using chromname for model naming)
    model = loopnete.loopnet(
        learning_rate=learning_rate,
        epochs=epochs,
        train_x=train_x_3d,
        train_y=train_labels,
        test_x=val_x_3d,  # Use validation set for monitoring
        test_y=val_labels,
        chromname=chromname,  # Use passed chromname
        kernel_size=kernel_size,
        save_dir=Aoutput
    )
    model.train_model()

def main(learning_rate, epochs, kernel_size, Apath, Aoutput, Abedpe, test_chrom, save_path, Aresolution, Awidth, Abalance):      
    warnings.filterwarnings("ignore")
    '''
        Parameter settings - Added negative sample ratio parameter
    '''
    np.seterr(divide='ignore', invalid='ignore')
    pathlib.Path(Aoutput).mkdir(parents=True, exist_ok=True)
    hic_info = utils.read_hic_header(Apath)
    if hic_info is None:
        hic = False
    else:
        hic = True

    coords = trainUtils.parsebed(Abedpe, lower=2, res=Aresolution)
    kde, lower, long_start, long_end = trainUtils.get_kde(coords)

    if not hic:
        import cooler
        Lib = cooler.Cooler(Apath)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = utils.get_hic_chromosomes(Apath, Aresolution)

    # Remove last two elements
    if len(chromosomes) > 2:
        chromosomes.pop()
        chromosomes.pop()
    
    positive_class = {}
    negative_class = {}
    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        
        if chromname not in coords:
            print(f'Skipping chromosome {chromname}, no coordinates')
            continue
            
        print('Reading data: {}'.format(key))
        if not hic:
            X = Lib.matrix(balance=Abalance,sparse=True).fetch(key).tocsr()
        else:
            if Abalance:
                X = utils.csr_contact_matrix('KR', Apath, key, key, 'BP', Aresolution)
            else:
                X = utils.csr_contact_matrix('NONE', Apath, key, key, 'BP', Aresolution)
        
        clist = coords[chromname]
        
        try:
            # Generate window samples for positive classes
            positive_class[chromname] = np.vstack((f for f in trainUtils.build_vector(X, clist, width=Awidth))) 
            
            neg_coords = trainUtils.negative_generating(X, kde, clist, lower, long_start, long_end)
            
            negative_class[chromname] = np.vstack((f for f in trainUtils.build_vector(X, neg_coords, width=Awidth, positive=False, )))

        except:
            print(chromname, ' Data read failed, please check data content.')
    
    # Prepare training dataset (excluding chr15)
    train_X, train_F = None, None
    test_X, test_y = None, None

    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr' + key
    
        if chromname not in positive_class:
            continue
    
        # Process current chromosome samples
        pos_data = positive_class[chromname]
        neg_data = negative_class[chromname]
    
        # Ensure negative samples don't exceed 3x positive samples
        n_neg = min(neg_data.shape[0], 3 * pos_data.shape[0])
        if neg_data.shape[0] > n_neg:
            idx = np.random.choice(neg_data.shape[0], n_neg, replace=False)
            neg_data = neg_data[idx]
    
        # Separate training and test sets
        if chromname == test_chrom:
            # Prepare test set data and labels (excluded from training)
            test_X = np.vstack((pos_data, neg_data))
            test_y = np.hstack((np.ones(pos_data.shape[0]), np.zeros(neg_data.shape[0])))
            print(f"Test chromosome {test_chrom} - Positive samples: {pos_data.shape[0]}, Negative samples: {n_neg}")
        else:
            # Accumulate training data
            if train_X is None:
                train_X = pos_data
                train_F = neg_data
            else:
                train_X = np.vstack((train_X, pos_data))
                train_F = np.vstack((train_F, neg_data))
            print(f"Training chromosome {chromname} - Positive samples: {pos_data.shape[0]}, Negative samples: {n_neg}")

    
    print(f"\nStarting model training - Training set positive samples: {train_X.shape[0]}, Negative samples: {train_F.shape[0]}")
    train(train_X, train_F, test_chrom, learning_rate, epochs, kernel_size, Aoutput)  # Pass hyperparameters
    print("Training completed")

    # Save test set
    joblib.dump((test_X, test_y), save_path)
    print(f"Model and test set saved, can be independently tested in new pipeline {test_chrom}")
    print(test_chrom, 'Training completed')

if __name__ == "__main__":

    learning_rate = 0.001
    epochs = 100
    kernel_size = 5
    path = "/public_data/yanghao/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
    output = "/home/yanghao/Loopnetae/model-chr15/"
    bedpe = "/home/yanghao/Loopnetae/data/gm12878_ctcf_h3k27ac.bedpe"
    # Define test chromosome and training chromosomes (all others)
    test_chrom = 'chr15'
    # Test set save path
    save_path = f'/home/yanghao/Loopnetae/testset/{test_chrom}.pkl'
    resolution = 10000
    width = 5
    balance = 1
    
    main(learning_rate, epochs, kernel_size, path, output, bedpe, test_chrom, save_path, resolution, width, balance)