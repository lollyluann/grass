import numpy as np

def group_metrics(y_true, y_pred, y_protected, label_protected=0, label_good=0):
    idx_prot = np.where(y_protected == label_protected)[0]
    idx_priv = np.where(y_protected != label_protected)[0]
    idx_good_class = np.where(y_true == label_good)[0]
    idx_pred_good_class = np.where(y_pred == label_good)[0]
    idx_bad_class = np.where(y_true != label_good)[0]
    idx_pred_bad_class = np.where(y_pred != label_good)[0]
    correct = y_true==y_pred
	
    TPR_prot = correct[np.intersect1d(idx_good_class, idx_prot)].mean()
    FP_prot = (1-correct[np.intersect1d(idx_pred_good_class, idx_prot)]).sum()
    FPR_prot = FP_prot/len(np.intersect1d(idx_bad_class, idx_prot))
    TPR_priv = correct[np.intersect1d(idx_good_class, idx_priv)].mean()
    FP_priv = (1-correct[np.intersect1d(idx_pred_good_class, idx_priv)]).sum()
    FPR_priv = FP_priv/len(np.intersect1d(idx_bad_class, idx_priv))
    
    accuracy = correct.mean()
    print('Accuracy is %f' % accuracy)
    
    bal_acc = (correct[idx_good_class].mean() + correct[idx_bad_class].mean())/2
    print('Balanced accuracy is %f' % bal_acc)
    
    # TPR_bad_prot = correct[np.intersect1d(idx_bad_class, idx_prot)].mean()
    # TPR_bad_priv = correct[np.intersect1d(idx_bad_class, idx_priv)].mean()
    gaps = np.array([np.abs(TPR_prot - TPR_priv), np.abs(FPR_prot - FPR_priv)])
    gap_rms = np.sqrt((gaps**2).mean())
    mean_gap = gaps.mean()
    max_gap = gaps.max()
    print('Gap RMS is', gap_rms)
    print('Mean absolute gap is', mean_gap)
    print('Max gap is', max_gap)
    
    average_odds_difference = ((TPR_prot - TPR_priv) + (FPR_prot - FPR_priv))/2
    print('Average odds difference is %f' % average_odds_difference)
    
    equal_opportunity_difference = TPR_prot - TPR_priv
    print('Equal opportunity difference is %f' % equal_opportunity_difference)
    
    statistical_parity_difference = (y_pred[idx_prot]==label_good).mean() - (y_pred[idx_priv]==label_good).mean()
    print('Statistical parity difference is %f' % statistical_parity_difference)

    return  accuracy, bal_acc, \
            gap_rms, mean_gap, max_gap, \
            average_odds_difference, equal_opportunity_difference, statistical_parity_difference
            
            
def accuracy_trace(y_true, y_pred, y_protected, verbose=True, label_protected=0):
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
     
    idx_minor = np.where(y_protected == label_protected)[0]
    idx_major = np.where(y_protected != label_protected)[0]
    correct = y_true==y_pred
    
    acc_major = correct[idx_major].mean()
    acc_minor = correct[idx_minor].mean()
    
    p_minor = np.linspace(0., 1., 11)
    p_major = 1 - p_minor
    acc = acc_minor*p_minor + acc_major*p_major
    
    if verbose:
        print(p_minor)
        print(acc)
        
    return acc, p_minor
    
    
