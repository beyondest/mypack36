import torch
import torchvision
import numpy as np
import time
from ..os_op.global_logger import *
from ..os_op.basic import *



class Yolov5_Post_Processor_Params(Params):
    def __init__(self,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 classes=None,
                 agnostic=False,
                 multi_label=False,
                 labels=(),
                 max_det=300) -> None:
        
        super().__init__()
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None             # [11] means only classes 11 are considered
        self.agnostic = False           # if True, then net only consider if there is something in the image, not the class
        self.multi_label = False        # if True, then one box can have multiple labels
        self.labels = ()                # labels to use for autolabelling ???
        self.max_det = 300              # maximum number of detections per image
    
        


def _box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def _xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def _nms(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
)->list:
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output


    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box/Mask
        box = _xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only

            conf,j= x[:, 5:mi].max(1,keepdim=True)
            
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = _box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            lr1.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def _bbox_to_corners(bbox):
    """
    Convert YOLOv5 bounding box to four corner points.

    Args:
        bbox (list): List containing four values representing the bounding box.

    Returns:
        list: List of four tuples representing the four corner points of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # Clockwise starting from top-left
    return corners



class Yolov5_Post_Processor:
    def __init__(self,
                 class_info:dict,
                 conf_thres:float = 0.25,
                 iou_thres:float = 0.45,
                 classes:list = None,
                 agnostic:bool = False,
                 multi_label:bool = False,
                 labels:list = (),
                 max_det:int = 300,
                 mode:str = 'Dbg') -> None:
        
        self.class_info = class_info
        self.params = Yolov5_Post_Processor_Params(conf_thres,iou_thres,classes,agnostic,multi_label,labels,max_det)

        CHECK_INPUT_VALID(mode,'Dbg','Rel')
        self.mode = mode
        
    @timing(1)
    def get_output(self, model_output0):
        """@timing(1)

        Args:
            model_output0 (_type_): _description_
            img_for_draw (Union[np.ndarray,None], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            (conts_list,probs_list,cls_name_list)
        """
        conts_list = []
        probs_list = []
        cls_name_list = []
        
        if model_output0 is None:
            lr1.error("Yolov5_Post_Processor: model_output is None, post processing failed, return [],[],[].")
            return (conts_list,probs_list,cls_name_list)
        if isinstance(model_output0,np.ndarray):
            model_output0 = torch.from_numpy(model_output0)
            
            #yolov5 model_output0.shape = (batch_size, total_num_anchors, nc + 5)
            
            nms_result = _nms(model_output0,
                              conf_thres=self.params.conf_thres,
                              iou_thres=self.params.iou_thres,
                              classes=self.params.classes,
                              agnostic=self.params.agnostic,
                              multi_label=self.params.multi_label,
                              labels=self.params.labels,
                              max_det=self.params.max_det,
                              nm=0)
            
            # len(nms_result) = batch_size
            nms_result = nms_result[0] # only one image in batch
            
            
            for each_detection in nms_result:
                bbox,prob,cls = each_detection[:4].numpy(),each_detection[4].item(),each_detection[5].item()
                cont = _bbox_to_corners(bbox)
                cont = np.array(cont,dtype=np.int32)
                conts_list.append(cont)
                probs_list.append(prob)
                cls_name_list.append(self.class_info[cls])
                
            return (conts_list,probs_list,cls_name_list)

        else:
            
            raise NotImplementedError("Yolov5_Post_Processor: model_output is not numpy array, post processing failed, return [],[],[].")
            
        

