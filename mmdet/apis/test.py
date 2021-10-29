# Copyright (c) OpenMMLab. All rights reserved.
from enum import Flag
import numpy as np
import torch
from cv2 import add
from mmdet.core.evaluation.class_names import voc_classes
import os.path as osp
import pickle
import shutil
import tempfile
import time
import json
import os
from xml.dom.minidom import Document
from mmdet.core import multiclass_nms

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.7,
                    return_meta=False
                    ):
    dy_th=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.7, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.7, 0.5, 0.5, 0.5, 0.5,
               0.5]
    model.eval()
    add_num = [0] * 20
    pseudo_num = [0] * 20
    ori_num = [0] * 20
    add_num_local = 0
    results = []
    return_img_metas = []
    return_boxes = []
    return_labels = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        batch_img_metas = data['img_metas'][0].data
        return_img_metas.extend(batch_img_metas[0])
        flag = False
        if 'gt_labels' in data.keys():
            tags = data.pop('gt_labels')
            box = data.pop('gt_bboxes')
            # ori_box = data.pop('ori_boxes')
            flag = True
        with torch.no_grad():
            result, det_box, det_label = model(return_loss=False, rescale=True, **data)
            # if flag:
            #     cur_add_num, cur_pseudo_num = gen_voc_label(data, result, tags, box, box, [show_score_thr] * len(VOC_CLASSES))
        batch_size = len(result)
        add_boxes = 0
        add_num_local += add_boxes
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)
        return_boxes.extend(det_box)
        return_labels.extend(det_label)

        for _ in range(batch_size):
            prog_bar.update()

    if return_meta:
        return results, return_img_metas, return_boxes, return_labels
    else:
        return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, pseudo_gen=False, annfile='filename'):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    return_img_metas = []
    return_boxes = []
    return_labels = []
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        batch_img_metas = data['img_metas'][0].data
        return_img_metas.extend(batch_img_metas[0])
        with torch.no_grad():
            if 'gt_labels' in data.keys():
                tags = data.pop('gt_labels')
                box = data.pop('gt_bboxes')
            # ori_box = data.pop('ori_boxes')
            result, det_box, det_label = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)
        return_boxes.extend(det_box)
        return_labels.extend(det_label)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        reduce_results = collect_results_gpu(results, len(dataset))
    else:
        reduce_results = collect_results_cpu(results, len(dataset), tmpdir)

    if pseudo_gen:
        with open(annfile, 'r') as load_f:
            load_dict = json.load(load_f)
            load_dict['annotations'] = []
            id = 0
            for i, box, label in zip(return_img_metas, return_boxes, return_labels):
                for b, l in zip(box, label):
                    ann = {}
                    ann['image_id'] = int(i['ori_filename'].split('.')[0])
                    ann['bbox'] = list(map(int, b[0:4].cpu().numpy().tolist()))
                    ann['category_id'] = l.item()
                    ann['id'] = id
                    ann['area'] = (ann['bbox'][2] - ann['bbox'][0]) * (ann['bbox'][3] - ann['bbox'][1])
                    id += 1
                    load_dict['annotations'].append(ann)
        wfile = annfile.split('.')[0] + str(rank) + '.' + annfile.split('.')[1]
        print('pseudo label save in ', wfile)
        json.dump(load_dict, open(wfile, 'w'))
    
    
    return reduce_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def gen_voc_label(data, result, tags, boxes, ori_boxes, pseudo_th=0.9):
    
    add_num = [0]*20
    pseudo_num = [0]*20
    for idx, (d, tags_img, box_img, ori_box, r) in enumerate(zip(data['img_metas'], tags, boxes, ori_boxes, result)):
        if 'test' in d.data[0][0]['filename']:
            continue
        doc = Document()
        annotation = doc.createElement("annotation")
        doc.appendChild(annotation)
        folder = doc.createElement("folder")
        filename = doc.createElement("filename")
        size = doc.createElement("size")
        width = doc.createElement("width")
        height = doc.createElement("height")
        depth = doc.createElement("depth")
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        annotation.appendChild(folder)
        annotation.appendChild(filename)
        annotation.appendChild(size)
        filenamee = doc.createTextNode(d.data[0][0]['ori_filename'].split('\\')[1])
        foldername = doc.createTextNode(d.data[0][0]['ori_filename'].split('\\')[0])
        widthnum = doc.createTextNode(str(d.data[0][0]['ori_shape'][0]))
        heightnum = doc.createTextNode(str(d.data[0][0]['ori_shape'][1]))
        depthnum = doc.createTextNode(str(d.data[0][0]['ori_shape'][2]))
        folder.appendChild(foldername)
        filename.appendChild(filenamee)
        width.appendChild(widthnum)
        height.appendChild(heightnum)
        depth.appendChild(depthnum)
        for t in tags_img[0]:
            tag = doc.createElement("tag")
            tagname = doc.createTextNode(VOC_CLASSES[t])
            tag.appendChild(tagname)
            annotation.appendChild(tag)
        # 补框代码
        # for b, t in zip(ori_box[0], tags_img[0]):
        #     b = b.numpy()
        #     point = doc.createElement("point")
        #     point_x = doc.createElement("point_x")
        #     point_y = doc.createElement("point_y")
        #     x = doc.createTextNode(str(((b[0] + b[2]) / 2))[7:-1])
        #     y = doc.createTextNode(str(((b[1] + b[3]) / 2))[7:-1])
        #     point_x.appendChild(x)
        #     point_y.appendChild(y)
        #     point.appendChild(point_x)
        #     point.appendChild(point_y)
        #     annotation.appendChild(point)
        #     flag = True
        #     for ridx in range(len(r)):
        #         for r_box in r[ridx]:
        #             if cal_iou(b,r_box[:4]) > 0.3 and r_box[4] > pseudo_th[ridx]:
        #                 flag = False
        #     if flag:
        #         add_num[t.item()] += 1
        #         objectt = doc.createElement("object")
        #         annotation.appendChild(objectt)
        #         bndbox = doc.createElement("bndbox")
        #         objectt.appendChild(bndbox)
        #         xmin = doc.createElement("xmin")
        #         ymin = doc.createElement("ymin")
        #         xmax = doc.createElement("xmax")
        #         ymax = doc.createElement("ymax")
        #         score = doc.createElement("score")
        #         calss = doc.createElement("name")
        #         objectt.appendChild(calss)
        #         calss.appendChild(doc.createTextNode(VOC_CLASSES[t]))
        #         bndbox.appendChild(xmin)
        #         bndbox.appendChild(ymin)
        #         bndbox.appendChild(xmax)
        #         bndbox.appendChild(ymax)
        #         bndbox.appendChild(score)
        #         difficult = doc.createElement("difficult")
        #         difficult.appendChild(doc.createTextNode('0'))
        #         objectt.appendChild(difficult)
        #         xminnum = doc.createTextNode(str(b[0]))
        #         yminnum = doc.createTextNode(str(b[1]))
        #         xmaxnum = doc.createTextNode(str(b[2]))
        #         ymaxnum = doc.createTextNode(str(b[3]))
        #         scorenum = doc.createTextNode(str(1))
        #         xmin.appendChild(xminnum)
        #         ymin.appendChild(yminnum)
        #         xmax.appendChild(xmaxnum)
        #         ymax.appendChild(ymaxnum)
        #         score.appendChild(scorenum)
        for ridx in range(len(r)):
            classname = VOC_CLASSES[ridx]
            for box in r[ridx]:
                # print(box[4])
                if box[4] < pseudo_th[ridx]:
                    continue
                # print(box)
                pseudo_num[t.item()] += 1
                objectt = doc.createElement("object")
                annotation.appendChild(objectt)
                bndbox = doc.createElement("bndbox")
                objectt.appendChild(bndbox)
                xmin = doc.createElement("xmin")
                ymin = doc.createElement("ymin")
                xmax = doc.createElement("xmax")
                ymax = doc.createElement("ymax")
                score = doc.createElement("score")
                calss = doc.createElement("name")
                objectt.appendChild(calss)
                calss.appendChild(doc.createTextNode(classname))
                bndbox.appendChild(xmin)
                bndbox.appendChild(ymin)
                bndbox.appendChild(xmax)
                bndbox.appendChild(ymax)
                bndbox.appendChild(score)
                difficult = doc.createElement("difficult")
                difficult.appendChild(doc.createTextNode('0'))
                objectt.appendChild(difficult)
                xminnum = doc.createTextNode(str(box[0]))
                yminnum = doc.createTextNode(str(box[1]))
                xmaxnum = doc.createTextNode(str(box[2]))
                ymaxnum = doc.createTextNode(str(box[3]))
                scorenum = doc.createTextNode(str(box[4]))
                xmin.appendChild(xminnum)
                ymin.appendChild(yminnum)
                xmax.appendChild(xmaxnum)
                ymax.appendChild(ymaxnum)
                score.appendChild(scorenum)
        f = open(os.path.join('C:/Users/Alex/WorkSpace/dataset/voc/VOCdevkit', os.path.join(d.data[0][0]['filename'].split('/')[7],
                                                             os.path.join('Annotations',
                                                                          d.data[0][0][
                                                                              'ori_filename'].split(
                                                                              '\\')[1].split('.')[
                                                                              0] + '.xml'))),
                 "w")
        f.write(doc.toprettyxml(indent="  "))
        f.close()
    return add_num, pseudo_num


def cal_iou(box1, box2):

    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    S_rec2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(box1[1], box2[1])
    right_line = min(box1[3], box2[3])
    top_line = max(box1[0], box2[0])
    bottom_line = min(box1[2], box2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
