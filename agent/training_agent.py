import os
import time

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from utils import get_logger, get_writer, set_random_seed, get_optimizer, get_lr_scheduler, resume_checkpoint, AverageMeter
from criterion import get_criterion
from dataflow import get_dataloader
from model import get_model_class

from .logging_tracker import LoggingTracker

class TrainingAgnet:
    def __init__(self, config, title):
        self.config = config

        self.logger = get_logger(config["logs_path"]["logger_path"])
        self.writer = get_writer(
            title,
            config["train"]["random_seed"],
            config["logs_path"]["writer_path"])

        if self.config["train"]["random_seed"] is not None:
            set_random_seed(config["train"]["random_seed"])

        self.device = torch.device(config["train"]["device"])

        self.train_loader, self.val_loader, self.test_loader, node_num, edge_indexs = get_dataloader(self.config["dataset"]["dataset"], 
                                                                self.config["dataset"]["dataset_path"], 
                                                                self.config["dataset"]["input_size"],
                                                                self.config["dataset"]["batch_size"], 
                                                                self.config["dataset"]["num_workers"], 
                                                                self.config["dataset"]["train_portion"],
                                                                self.config["dataset"]["slide_win"],
                                                                self.config["dataset"]["slide_stride"]
                                                                )


        model_class = get_model_class(self.config["agent"]["model_agent"])
        model = model_class(node_num, edge_indexs, self.config["model"], self.device)
        self.model = model.to(self.device)
        self.model = self._parallel_process(self.model)

        criterion = get_criterion(config["agent"]["criterion_agent"], config["criterion"])
        self.criterion = criterion.to(self.device)
        self.criterion = self._parallel_process(self.criterion)

        self.logging_tracker = LoggingTracker(self.writer)

        self._optimizer_init(self.model, self.criterion)

        self.epochs = config["train"]["epochs"]
        self.start_epochs = 0

        self.config = config

    def fit(self):
        """ Fit agent for searching or evaluating.
        """
        start_time = time.time()
        self.logger.info("Training process start!")

        self.train_loop()

    def train_loop(self):
        best_val_performance = -float("inf")

        for epoch in range(self.start_epochs, self.epochs):
            self.logger.info(f"Start to train for epoch {epoch}")
            self.logger.info(f"Learning Rate : {self.optimizer.param_groups[0]['lr']:.8f}")

            self._training_step(self.model, self.train_loader, epoch)
            #val_loss = self._validate_step(self.model, self.val_loader, epoch)

            self.evaluate(self.model, self.val_loader, self.test_loader, epoch)

            self.logging_tracker.record(epoch)

    def _training_step(self, model, train_loader, epoch, print_freq=20):
        losses = AverageMeter()

        model.train()
        start_time = time.time()
        
        for step, (X, y, label) in enumerate(train_loader):
            X, y, label = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)

            N = X.shape[0]
            self.optimizer.zero_grad()
            out = model(X)
            
            loss = self.criterion(out, y)
            loss.backward()

            self.logging_tracker.step(out, y)

            self.optimizer.step()

            losses.update(loss.item(), N)

            if (step > 1 and step % print_freq == 0) or (step == len(train_loader) - 1):
                self.logger.info(f"Train : [{(epoch+1):3d}/{self.epochs}] "
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f}")

        self.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)
        self.logger.info(
            f"Train: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f}" 
            f"Time {time.time() - start_time:.2f}")



    def _validate_step(self, model, val_loader, epoch, print_freq=20):
        losses = AverageMeter()

        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y, label) in enumerate(val_loader):
                X, y, label = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)
                N = X.shape[0]
                out = model(X)
                
                loss = self.criterion(out, y)
                losses.update(loss.item(), N)

                if (step > 1 and step % print_freq == 0) or (step == len(val_loader) - 1):
                    self.logger.info(f"Val: [{(epoch+1):3d}/{self.epochs}] "
                                     f"Step {step:3d}/{len(val_loader)-1:3d} Loss {losses.get_avg():.3f}")

            self.writer.add_scalar("Val/_loss/", losses.get_avg(), epoch)
            self.logger.info(
                f"Val: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f}" 
                f"Time {time.time() - start_time:.2f}")

        return losses.get_avg()


    def evaluate(self, model, val_loader, test_loader, epoch):
        self.logger.info("Evaluating starting ------------------")
        model.eval()
        start_time = time.time()

        val_predict_list, val_gt_list, val_label_list = self._evaluate_step(model, val_loader)
        test_predict_list, test_gt_list, test_label_list = self._evaluate_step(model, test_loader)

        val_err_scores = self._evaluate_err_score(val_predict_list, val_gt_list)
        test_err_scores = self._evaluate_err_score(test_predict_list, test_gt_list)

        anomaly_threshold = torch.max(val_err_scores)
        print(anomaly_threshold)

        val_predict_anomaly = val_err_scores > anomaly_threshold
        val_predict_anomaly = val_predict_anomaly.long().tolist()

        test_predict_anomaly = test_err_scores > anomaly_threshold
        test_predict_anomaly = test_predict_anomaly.long().tolist()

        val_label_list = val_label_list.tolist()

        test_label_list = test_label_list.tolist()
        test_f1_score = f1_score(test_label_list, test_predict_anomaly)
        test_recall_score = recall_score(test_label_list, test_predict_anomaly)
        test_precision_score = precision_score(test_label_list, test_predict_anomaly)

        self.writer.add_scalar("Test/_f1_score/", test_f1_score, epoch)
        self.writer.add_scalar("Test/_precision/", test_precision_score, epoch)
        self.writer.add_scalar("Test/_recall/", test_recall_score, epoch)
        self.logger.info(
            f"Test: [{epoch+1:3d}/{self.epochs}] Final F1 Score {test_f1_score}"
            f" Final Recall Score {test_recall_score} Final Precision Score {test_precision_score}"
            f" Time {time.time() - start_time:.2f}")
        self.logger.info("Evaluating End !")

    def _evaluate_step(self, model, loader):
        predict_list = []
        gt_list = []
        label_list = []
        with torch.no_grad():
            for step, (X, y, label) in enumerate(loader):
                X, y, label = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)

                out = model(X)

                predict_list.append(out.cpu().detach().clone())
                gt_list.append(y.cpu())
                label_list.append(label.cpu())

        predict_list = torch.cat(predict_list)
        gt_list = torch.cat(gt_list)
        label_list = torch.cat(label_list)

        return predict_list, gt_list, label_list

    def _evaluate_err_score(self, predict_list, gt_list, eps=1e-2):
        err_scores = torch.abs(gt_list - predict_list)

        err_scores_median = torch.median(err_scores, dim=0)[0]
        err_scores_quantile_1 = torch.quantile(err_scores, q=0.25, dim=0)
        err_scores_quantile_3 = torch.quantile(err_scores, q=0.75, dim=0)

        normalize_err_scores = (err_scores - err_scores_median) / (torch.abs(err_scores_quantile_3 - err_scores_quantile_1)+eps)

        err_score = torch.max(normalize_err_scores, dim=-1)[0]
        return err_score



    def _optimizer_init(self, model, criterion):
        self.optimizer = get_optimizer(
            [{"params": model.parameters()},
             {"params": criterion.parameters()}],
            self.config["optim"]["optimizer"],
            learning_rate=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["weight_decay"],
            logger=self.logger,
            momentum=self.config["optim"]["momentum"],
            alpha=self.config["optim"]["alpha"],
            beta=self.config["optim"]["beta"])

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.config["optim"]["scheduler"],
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.config["optim"]["decay_step"],
            decay_ratio=self.config["optim"]["decay_ratio"],
            total_epochs=self.config["train"]["epochs"])

    def _resume(self, model, criterion):
        """ Load the checkpoint of model, optimizer, and lr scheduler.
        """
        if self.config["train"]["resume"]:
            self.start_epochs = resume_checkpoint(
                    model,
                    self.config["experiment_path"]["resume_path"],
                    criterion,
                    None,
                    None)
            self.logger.info(
                "Resume training from {} at epoch {}".format(
                    self.config["experiment_path"]["resume_path"], self.start_epochs))

    def _parallel_process(self, model):
        if self.device.type == "cuda" and self.config["train"]["ngpu"] >= 1:
            return nn.DataParallel(
                model, list(range(self.config["train"]["ngpu"])))
        else:
            return model


    def _optimizer_init(self, model, criterion):
        self.optimizer = get_optimizer(
            [{"params": model.parameters()},
             {"params": criterion.parameters()}],
            self.config["optim"]["optimizer"],
            learning_rate=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["weight_decay"],
            logger=self.logger,
            momentum=self.config["optim"]["momentum"],
            alpha=self.config["optim"]["alpha"],
            beta=self.config["optim"]["beta"])

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.config["optim"]["scheduler"],
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.config["optim"]["decay_step"],
            decay_ratio=self.config["optim"]["decay_ratio"],
            total_epochs=self.config["train"]["epochs"])
