# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
from detectron.utils.vis import *

def vis_one_image_opencv(
        im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False):
    """Constructs a numpy array with the detections visualized."""

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show class (off by default)
        if show_class:
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], kp_thresh)

    return im


def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, body_uv=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf'):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints, _ = keypoint_utils.get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)
                
    #   DensePose Visualization Starts!!
    ##  Get full IUV image out 
    IUV_fields = body_uv[1]
    #
    All_Coords = np.zeros(im.shape)
    All_inds = np.zeros([im.shape[0],im.shape[1]])
    K = 26
    ##
    inds = np.argsort(boxes[:,4])
    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind,:]
        if entry[4] > 0.65:
            entry=entry[0:4].astype(int)
            ####
            output = IUV_fields[ind]
            ####
            All_Coords_Old = All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]
            All_Coords_Old[All_Coords_Old==0]=output.transpose([1,2,0])[All_Coords_Old==0]
            All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]= All_Coords_Old
            ###
            CurrentMask = (output[0,:,:]>0).astype(np.float32)
            All_inds_old = All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]]
            All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0]*i
            All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]] = All_inds_old
    #
    All_Coords[:,:,1:3] = 255. * All_Coords[:,:,1:3]
    All_Coords[All_Coords>255] = 255.
    All_Coords = All_Coords.astype(np.uint8)
    All_inds = All_inds.astype(np.uint8)
    #
    IUV_SaveName = os.path.basename(im_name).split('.')[0]+'_IUV.png'
    INDS_SaveName = os.path.basename(im_name).split('.')[0]+'_INDS.png'
    cv2.imwrite(os.path.join(output_dir, '{}'.format(IUV_SaveName)), All_Coords )
    cv2.imwrite(os.path.join(output_dir, '{}'.format(INDS_SaveName)), All_inds )
    print('IUV written to: ' , os.path.join(output_dir, '{}'.format(IUV_SaveName)) )
    ###
    ### DensePose Visualization Done!!
    #
    # output_name = os.path.basename(im_name) + '.' + ext
    # fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    # plt.close('all')

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        vid = im_name.split("/")[1]
        output_dir = args.output_dir + vid
        try:
            os.mkdir(output_dir)
        except OSError:
            pass
        out_name = os.path.join(
            output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        if os.path.exists(out_name):
            continue
        
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
