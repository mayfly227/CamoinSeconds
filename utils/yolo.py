from typing import List

import torch


class YoloRewardModel(torch.nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO reward training. Install it in the training environment first."
            ) from exc

        self.yolo = YOLO(weight_path)
        self.model = self.yolo.eval()

        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)

    def targetLoss(self, images: torch.Tensor):
        cls_idx = 1
        results_train = self.model(
            images, verbose=False, conf=0.000001, iou=0.5, embed=[-1]
        )
        loss = -results_train[0][:, 4 + cls_idx, :].sum()
        return True, loss

    def confidence_loss(
        self,
        images: torch.Tensor,
        mode: str = "non_targeted",
        target_class_id: int = 0,
    ) -> tuple[bool, torch.Tensor]:
        """Compute differentiable confidence-based loss from YOLO raw predictions.

        non_targeted: minimize sum of confidences over all classes.
        targeted: minimize confidence for one target class.
        """
        results_train = self.model(
            images, verbose=False, conf=0.000001, iou=0.5, embed=[-1]
        )
        class_scores = results_train[0][:, 4:, :]

        if mode == "non_targeted":
            loss = class_scores.sum()
        elif mode == "targeted":
            if target_class_id < 0 or target_class_id >= class_scores.shape[1]:
                raise ValueError(
                    f"target_class_id out of range: {target_class_id}, valid [0, {class_scores.shape[1] - 1}]"
                )
            loss = class_scores[:, target_class_id, :].sum()
        else:
            raise ValueError(
                f"Unknown YOLO reward mode: {mode}. Expected 'non_targeted' or 'targeted'."
            )

        return True, loss

    def maxProExtract(
        self,
        images: torch.Tensor,
        class_id: List[int],
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> tuple[bool, torch.Tensor]:
        device = images.device
        if len(class_id) == 0:
            class_id = list(range(80))
        results_train = self.model(images, verbose=False, conf=conf, iou=iou)
        loss1 = torch.FloatTensor([0.0]).to(device=device)
        for res1 in results_train:
            for box in res1.boxes:
                conf = box.conf
                clas = box.cls
                if len(conf) == 0:
                    continue
                for con, c in zip(conf, clas):
                    if con > 0 and c in class_id:
                        # if con > 0:
                        loss1 = loss1 + con
        if loss1.item() < 0.00001:
            return False, torch.FloatTensor([0.0]).to(device=device)
        return True, loss1

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "non_targeted",
        target_class_id: int = 0,
    ) -> tuple[bool, torch.Tensor]:
        return self.maxProExtract(
            images,
            [target_class_id])
        return self.confidence_loss(
            images,
            mode=mode,
            target_class_id=target_class_id,
        )
