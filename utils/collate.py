from utils.functions import *
from torch.utils.data._utils.collate import default_collate

def seq_collate_OnlineHDmap(data):
    """
    Custom collate function for BEV dataset that handles variable-length sequences.
    Keeps variable-length fields as lists (not stacked).
    """
    # Extract variable-length fields before default_collate tries to stack them
    polylines_list = [item['polylines'] for item in data] if 'polylines' in data[0] else None
    bbox_anns_list = [item['bbox_anns'] for item in data] if 'bbox_anns' in data[0] else None
    
    # Remove them temporarily to avoid stacking errors
    data_for_collate = []
    for item in data:
        item_copy = item.copy()
        if 'polylines' in item_copy:
            del item_copy['polylines']
        if 'bbox_anns' in item_copy:
            del item_copy['bbox_anns']
        data_for_collate.append(item_copy)
    
    # Use default collate for other fields
    batch = default_collate(data_for_collate)
    
    # Add back the variable-length sequences as lists
    if polylines_list is not None:
        batch['polylines'] = polylines_list
    if bbox_anns_list is not None:
        batch['bbox_anns'] = bbox_anns_list
    
    return batch

def seq_collate_BEV(data):
    """
    Custom collate function for BEV dataset that handles variable-length sequences.
    Keeps variable-length fields as lists (not stacked).
    """
    # Extract variable-length fields before default_collate tries to stack them
    input_ids_list = [item['input_ids'] for item in data] if 'input_ids' in data[0] else None
    vlm_labels_list = [item['vlm_labels'] for item in data] if 'vlm_labels' in data[0] else None
    polylines_list = [item['polylines'] for item in data] if 'polylines' in data[0] else None
    
    # Remove them temporarily to avoid stacking errors
    data_for_collate = []
    for item in data:
        item_copy = item.copy()
        if 'input_ids' in item_copy:
            del item_copy['input_ids']
        if 'vlm_labels' in item_copy:
            del item_copy['vlm_labels']
        if 'polylines' in item_copy:
            del item_copy['polylines']
        data_for_collate.append(item_copy)
    
    # Use default collate for other fields
    batch = default_collate(data_for_collate)
    
    # Add back the variable-length sequences as lists
    if input_ids_list is not None:
        batch['input_ids'] = input_ids_list
    if vlm_labels_list is not None:
        batch['vlm_labels'] = vlm_labels_list
    if polylines_list is not None:
        batch['polylines'] = polylines_list
    
    return batch


