import matplotlib

matplotlib.use("Agg")
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import argparse
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from cv2 import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
import shutil
from torch.utils.tensorboard import SummaryWriter
from models.model import School
from dataset.MVTec import MVTecDataset, std_train, mean_train
from loguru import logger

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class IKD:
    def __init__(self, args):

        # extract experiment settings
        self.device = f'cuda:{args.gpu}'
        self.model = args.model
        self.save_path = args.save_path
        self.exp_name = args.exp_name
        self.category = args.category
        self.lr_rate = args.lr_rate
        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.cal_pro = args.cal_pro
        self.feature_choose = args.feature_choose
        self.feature_weight = args.feature_weight
        self.beta = args.beta
        self.gamma = args.gamma
        self.n_epoch = args.epoch
        self.validate_step = args.validate_step
        self.input_size = args.input_size
        self.save_images = args.save_images


        # prepare directories
        self.prepare_dirs()

        # build model
        self.model = School(self.model).to(self.device)

        # prepare transforms
        self.data_transforms = transforms.Compose([
            transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((args.load_size, args.load_size), Image.NEAREST),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),])
        self.inv_normalize = transforms.Normalize(mean=mean_train,
                                                  std=std_train)

    def prepare_dirs(self):
        # the root dir of current experiment
        self.root_dir = os.path.join(self.save_path, self.exp_name)
        logger.start(os.path.join(self.root_dir, f'{self.category}.log'))

        logger.info(f'root dir: {self.root_dir}')

        # the dir to save image samples
        self.sample_dir = os.path.join(self.root_dir, 'sample')

        # model path
        self.model_path = os.path.join(self.root_dir, 'models')

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def get_tensorboard_logger(self, reset_version=False):
        self.tensorboard_root_path = os.path.join(self.save_path, "tensorboard", self.exp_name,
                                                  self.category)
        os.makedirs(self.tensorboard_root_path, exist_ok=True)

        if reset_version:
            shutil.rmtree(os.path.join(self.tensorboard_root_path))

        # identify version
        i = 0
        while os.path.exists(os.path.join(self.tensorboard_root_path, f'version_{i}')):
            i = i + 1

        self.tensorboard_path = os.path.join(self.save_path, "tensorboard", self.exp_name,
                                             self.category, f'version_{i}')

        return SummaryWriter(log_dir=self.tensorboard_path)

    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lr_rate)

    def get_train_loader(self):
        logger.info(f'get train loader category: {self.category}')
        image_datasets = MVTecDataset(root=os.path.join(self.dataset_path, self.category),
                                      transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0)  # , pin_memory=True)
        return train_loader

    def get_test_loader(self):
        logger.info(f'get test loader category: {self.category}')
        test_datasets = MVTecDataset(root=os.path.join(self.dataset_path, self.category),
                                     transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False,
                                 num_workers=0)  # , pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.category}-model.pt'))


    def cal_auc(self, score_map_list, test_mask_list):

        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        pixel_level_ROCAUC = roc_auc_score(flatten_mask_list, flatten_score_map_list)

        if self.cal_pro:
            pro_auc_score = self.cal_pro_metric(test_mask_list, score_map_list, fpr_thresh=0.3)
        else:
            pro_auc_score = 0

        return pixel_level_ROCAUC, pro_auc_score

    def cal_pro_metric(self, labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
        labeled_imgs[labeled_imgs <= 0.45] = 0
        labeled_imgs[labeled_imgs > 0.45] = 1
        labeled_imgs = labeled_imgs.astype(np.bool)

        max_th = score_imgs.max()
        min_th = score_imgs.min()
        delta = (max_th - min_th) / max_steps

        ious_mean = []
        ious_std = []
        pros_mean = []
        pros_std = []
        threds = []
        fprs = []
        binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
        for step in range(max_steps):
            thred = max_th - step * delta
            # segmentation
            binary_score_maps[score_imgs <= thred] = 0
            binary_score_maps[score_imgs > thred] = 1

            pro = []  # per region overlap
            iou = []  # per image iou
            # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
            # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
            for i in range(len(binary_score_maps)):  # for i th image
                # pro (per region level)
                label_map = measure.label(labeled_imgs[i], connectivity=2)
                props = measure.regionprops(label_map)
                for prop in props:
                    x_min, y_min, x_max, y_max = prop.bbox
                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                    cropped_mask = prop.filled_image  # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                    pro.append(intersection / prop.area)
                # iou (per image level)
                intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
                union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
                if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                    iou.append(intersection / union)
            # against steps and average metrics on the testing data
            ious_mean.append(np.array(iou).mean())
            ious_std.append(np.array(iou).std())
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
            # fpr for pro-auc
            masks_neg = ~labeled_imgs
            fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
            fprs.append(fpr)
            threds.append(thred)

        # as array
        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        pros_std = np.array(pros_std)
        fprs = np.array(fprs)

        # default 30% fpr vs pro, pro_auc
        idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
        fprs_selected = fprs[idx]
        fprs_selected = self.rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
        pros_mean_selected = pros_mean[idx]
        pro_auc_score = auc(fprs_selected, pros_mean_selected)
        return pro_auc_score

    def rescale(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def plot_sample(self, names, imgs, anomalies, gts):
        # get subplot number
        subplot_number = len(anomalies) + 1
        total_number = len(imgs)

        # normarlisze anomalies
        nmax = 0
        nmin = 1e8

        for dict_v in anomalies.values():
            for v in dict_v:
                nmax = max(nmax, np.max(v))
                nmin = min(nmin, np.min(v))

        for key in anomalies:
            for i in range(len(anomalies[key])):
                anomalies[key][i] = (anomalies[key][i] - nmin) / (nmax - nmin)

        # draw gts
        mask_imgs = []
        for idx in range(total_number):
            gts_ = gts[idx]
            mask_imgs_ = imgs[idx]
            mask_imgs_[gts_ > 0.5] = (255, 0, 0)
            mask_imgs.append(mask_imgs_)

        # save imgs
        for idx in range(total_number):
            plt.figure()
            plt.clf()
            plt.subplot(1, subplot_number, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('GT')
            plt.imshow(mask_imgs[idx])

            n = 2
            for key in anomalies:
                plt.subplot(1, subplot_number, n)
                plt.xticks([])
                plt.yticks([])
                plt.title(key)

                # display max
                plt.imshow(anomalies[key][idx], cmap="jet")
                n = n + 1

            plt.savefig(f"{self.sample_dir}/{names[idx]}.jpg", bbox_inches='tight', dpi=300)
            plt.close()

    def metric_cal(self, scores, gt_list, gt_mask_list):
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list, dtype=int)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)

        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list, dtype=int)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

        return img_roc_auc, per_pixel_rocauc, threshold

    def calculate_loss(self, x_t, x_s, margin_last, margin):
        loss = {}

        for key in self.feature_choose:
            x_t_ = x_t[key]
            x_s_ = x_s[key]

            BS, C, H, W = x_t_.shape

            # normalize
            x_t_ = F.normalize(x_t_, p=2, dim=1)
            x_s_ = F.normalize(x_s_, p=2, dim=1)

            ####### AHSM
            dif = (x_t_ - x_s_) ** 2
            dif = torch.sum(dif, dim=1)

            # calculate the margin for hard samples mining
            mu = dif.mean().item()
            variance = dif.std().item()

            # exponential moving average
            margin_last = margin
            margin = 0.01 * margin_last + 0.99 * (mu + self.beta * variance)

            select_idx = dif >= margin
            select_idx = select_idx.view(-1)

            ####### Pixel-wise similarity loss
            dif = dif[dif >= margin]
            l_ps = torch.mean(dif)

            ###### Context similarity loss
            Qt = x_t_.permute(0, 2, 3, 1).reshape(-1, C)
            Qs = x_s_.permute(0, 2, 3, 1).reshape(-1, C)

            Qt = Qt[select_idx, :]
            Qs = Qs[select_idx, :]

            Gt = Qt @ Qt.T
            Gs = Qs @ Qs.T

            l_cs = (Gt - Gs) ** 2
            l_cs = l_cs.mean()

            loss[key] = l_ps + self.gamma * l_cs

        loss_mean = self.sum_dict(loss)
        loss['mean'] = loss_mean
        return loss, margin, margin_last

    def denormalization(self, x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

        return x

    def sum_dict(self, x):
        weight = self.feature_weight
        x_sum = weight[0] * x[self.feature_choose[0]]
        for i in range(1, len(self.feature_choose)):
            x_sum += weight[i] * x[self.feature_choose[i]]
        return x_sum / sum(weight)

    def mul_dict(self, x):
        x_sum = x[self.feature_choose[0]]
        for i in range(1, len(self.feature_choose)):
            x_sum = x[self.feature_choose[i]] * x_sum
        return x_sum

    def calculate_anomaly_map(self, x_t, x_s):
        ams = {}
        features = self.feature_choose

        for idx in range(len(features)):
            x_t_ = x_t[features[idx]]
            x_s_ = x_s[features[idx]]

            # normalize
            x_t_ = F.normalize(x_t_, p=2, dim=1)
            x_s_ = F.normalize(x_s_, p=2, dim=1)

            # L2
            dif = (x_t_ - x_s_) ** 2
            am = torch.sum(dif, dim=1)

            ams[features[idx]] = am

        ams['mean'] = self.sum_dict(ams)
        ams['mul'] = self.mul_dict(ams)

        return ams

    def train(self):
        # initial step
        step = 0

        # get trainloader
        train_loader = self.get_train_loader()

        # get testloader
        test_loader = self.get_test_loader()

        # optimizer
        optimizer = self.get_optimizer(self.model.student)

        # tensorboard_logger
        tensorboard_logger = self.get_tensorboard_logger(reset_version=True)

        self.model.teacher.eval()
        self.model.student.train()

        margin_last = 0
        margin = 0
        for epoch in range(self.n_epoch):

            for (x, _, _, file_name, _) in train_loader:
                x = x.to(self.device)

                t_out, s_out = self.model(x)  # get the model's outputs

                loss, margin, margin_last = \
                    self.calculate_loss(t_out, s_out, margin_last=margin_last, margin=margin)

                optimizer.zero_grad()
                loss['mean'].backward()
                optimizer.step()

                tensorboard_logger.add_scalar(f"Train/margin", margin, step)

                for k, v in loss.items():
                    tensorboard_logger.add_scalar(f"Train/{k}", v.detach().cpu(), step)

                step += 1

            # test
            if epoch % self.validate_step == 0 and epoch != 0 or epoch > self.n_epoch - 5:
                if epoch == self.n_epoch - 1:
                    save_images = self.save_images
                else:
                    save_images = False

                pixel_level_rocauc, pro_auc_score = self.test(test_loader, save_images=save_images)
                self.save_model()

                # change into training module
                self.model.teacher.eval()
                self.model.student.train()

                # logger
                for key in pixel_level_rocauc:
                    tensorboard_logger.add_scalar(f"Val/pixel_level_rocauc/{key}", pixel_level_rocauc[key], epoch)
                    tensorboard_logger.add_scalar(f"Val/pro_auc_score/{key}", pro_auc_score[key], epoch)

                    logger.info(f'{self.category}:{epoch}/{self.n_epoch}, {key}: pxl auc: {pixel_level_rocauc[key]} pro: {pro_auc_score[key]}')


    def test(self, test_loader, save_images):
        self.model.teacher.eval()
        self.model.student.eval()

        dsize = self.input_size, self.input_size
        predictions = None

        ground_truths = []
        ground_truths_image_level = []
        names = []
        imgs = []

        for (x, mask, y, file_name, x_type) in test_loader:
            x = x.to(self.device)

            t_o, s_o = self.model(x)

            anomaly_maps = self.calculate_anomaly_map(t_o, s_o)
            anomaly_maps = {k: v.detach().cpu().numpy() for k, v in anomaly_maps.items()}

            image = self.denormalization(x[0].detach().cpu().numpy())
            imgs.append(image)

            mask = mask.numpy()
            y = y.numpy()
            ground_truths.append(cv2.resize(mask[0, 0, :, :], dsize, interpolation=cv2.INTER_NEAREST))
            ground_truths_image_level.append(y)
            names.append(f'{self.category}-{x_type[0]}-{file_name[0]}')

            if predictions is None:
                predictions = {k:[] for k in anomaly_maps}

            for key in anomaly_maps:
                anomaly_score_ = anomaly_maps[key]
                predictions[key].append(cv2.resize(anomaly_score_[0, 0, :, :], dsize) if len(
                    anomaly_score_.shape) == 4 else cv2.resize(
                    anomaly_score_[0, :, :], dsize))

        if save_images:
            self.plot_sample(names, imgs, predictions, ground_truths)

        pixel_level_rocauc, pro_auc_score = {}, {}
        for key in predictions:

            pixel_level_rocauc[key] = []
            pro_auc_score[key] = []

            for key in predictions:
                pixel_level_rocauc[key], pro_auc_score[key] = self.cal_auc(
                    np.array(predictions[key]), np.array(ground_truths))


        return pixel_level_rocauc, pro_auc_score


def parse_args():
    parser = argparse.ArgumentParser()

    # path options
    parser.add_argument('--dataset_path', default=r'../datasets/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument('--exp_name', type=str, required=False,default="exp_default")

    # training options
    parser.add_argument('--gpu', type=int, required=False, default=0)
    parser.add_argument('--epoch', type=int, required=False, default=50)
    parser.add_argument('--lr_rate', type=float, required=False, default=0.00001)
    parser.add_argument('--batch_size', type=int, required=False, default=2)

    parser.add_argument('--load_size', type=int, required=False, default=256)
    parser.add_argument('--input_size', type=int, required=False, default=256)
    parser.add_argument('--category', type=str, default='tile')
    parser.add_argument('--validate_step', type=int, default=10)

    parser.add_argument('--save_images', type=str2bool, default=True)
    parser.add_argument('--cal_pro', type=str2bool, default=True)


    # hyper-parameters
    parser.add_argument('--feature_choose', type=str, nargs='+',default=['x1', 'x2'])
    parser.add_argument('--feature_weight', type=float, default=[1., 2.])
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--beta', type=float, default=2)

    # backbone
    parser.add_argument('--model', type=str, choices=['hrnet18', 'hrnet32', 'hrnet48'], default='hrnet32')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    trainer = IKD(args)
    trainer.train()
