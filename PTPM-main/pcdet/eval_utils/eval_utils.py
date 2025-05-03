def filter_eval_boxes(pred_dicts, score_threshold):
    """
    Filter boxes based on score threshold
    """
    for key in pred_dicts.keys():
        if key == 'pred_boxes':
            # Convert float64 to float32 if needed
            if pred_dicts[key].dtype == np.float64:
                pred_dicts[key] = pred_dicts[key].astype(np.float32)
            mask = pred_dicts['pred_scores'] > score_threshold
            pred_dicts[key] = pred_dicts[key][mask]
        elif key == 'pred_scores':
            # Convert float64 to float32 if needed
            if pred_dicts[key].dtype == np.float64:
                pred_dicts[key] = pred_dicts[key].astype(np.float32)
            mask = pred_dicts[key] > score_threshold
            pred_dicts[key] = pred_dicts[key][mask]
        elif key == 'pred_labels':
            mask = pred_dicts['pred_scores'] > score_threshold
            pred_dicts[key] = pred_dicts[key][mask]
    return pred_dicts 