import os

import torch

from utils.references import METRIC_ACCURACY, METRIC_F1, METRIC_LOSS, METRIC_PRECISION, METRIC_RECALL


def tokenize_float(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}".replace(".", "p")


def print_section(title: str, items: dict[str, object], width: int = 84) -> None:
    line = "=" * width
    print(line)
    print(title)
    print("-" * width)
    for key, value in items.items():
        print(f"{key:32}: {value}")
    print(line)


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mbatch_loss_group: int = -1,
    progress_label: str | None = None,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: torch.amp.GradScaler | None = None,
):
    net.train()
    running_loss = 0.0
    mbatch_losses = []
    grad_accum_steps = max(1, int(grad_accum_steps))
    use_amp_now = bool(use_amp and device.type == "cuda")
    if use_amp_now and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
    progress_bar = None
    total_batches = len(train_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label=progress_label)
    optimizer.zero_grad(set_to_none=True)
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp_now):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        loss_for_backward = loss / grad_accum_steps
        if use_amp_now and scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((i + 1) % grad_accum_steps == 0) or (i + 1 == total_batches)
        if should_step:
            if use_amp_now and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        if mbatch_loss_group > 0 and (i + 1) % mbatch_loss_group == 0:
            mbatch_losses.append(running_loss / mbatch_loss_group)
            running_loss = 0.0
        if progress_bar:
            progress_bar.increment()
    if progress_bar:
        progress_bar.finish()
    if mbatch_loss_group > 0:
        return mbatch_losses


def _compute_weighted_multilabel_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    tps = (predictions * labels).sum(dim=0)
    fps = (predictions * (1.0 - labels)).sum(dim=0)
    class_total = labels.sum(dim=0)

    prec_denom = tps + fps
    class_prec = torch.where(prec_denom > 0, tps / prec_denom, torch.zeros_like(tps))
    class_recall = torch.where(class_total > 0, tps / class_total, torch.zeros_like(tps))

    freqs = class_total
    inv_freq = torch.where(freqs > 0, 1.0 / freqs, torch.zeros_like(freqs))
    inv_freq_sum = inv_freq.sum()
    if inv_freq_sum > 0:
        class_weights = inv_freq / inv_freq_sum
    else:
        class_weights = torch.full((num_classes,), 1.0 / num_classes, device=freqs.device)

    precision = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    if precision > 0 and recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = torch.tensor(0.0, device=precision.device)

    total_positives = freqs.sum()
    accuracy = torch.where(total_positives > 0, tps.sum() / total_positives, torch.tensor(0.0, device=tps.device))
    return {
        METRIC_ACCURACY: accuracy,
        METRIC_F1: f1,
        METRIC_PRECISION: precision,
        METRIC_RECALL: recall,
    }


def validation_loop(
    val_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    num_classes: int,
    device: torch.device,
    multi_label: bool = True,
    th_multi_label: float = 0.5,
    one_hot: bool = False,
    class_metrics: bool = False,
    progress_label: str | None = None,
    apply_sigmoid: bool = False,
):
    net.eval()
    loss = 0.0
    size = len(val_loader.dataset)
    class_total = torch.zeros(num_classes, dtype=torch.float32)
    class_tp = torch.zeros(num_classes, dtype=torch.float32)
    class_fp = torch.zeros(num_classes, dtype=torch.float32)
    progress_bar = None
    total_batches = len(val_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label=progress_label)

    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            scores = torch.sigmoid(outputs) if apply_sigmoid else outputs
            if not multi_label:
                predictions = torch.zeros_like(scores)
                predictions[torch.arange(scores.shape[0]), torch.argmax(scores, dim=1)] = 1.0
            else:
                predictions = torch.where(scores > th_multi_label, 1.0, 0.0)
            if not one_hot:
                labels_mat = torch.zeros_like(scores)
                labels_mat[torch.arange(scores.shape[0]), labels] = 1.0
                labels = labels_mat

            tps = predictions * labels
            fps = predictions - tps

            class_tp += tps.sum(dim=0).cpu()
            class_fp += fps.sum(dim=0).cpu()
            class_total += labels.sum(dim=0).cpu()
            if progress_bar:
                progress_bar.increment()
    if progress_bar:
        progress_bar.finish()

    prec_denom = class_tp + class_fp
    class_prec = torch.where(prec_denom > 0, class_tp / prec_denom, torch.zeros_like(class_tp))
    class_recall = torch.where(class_total > 0, class_tp / class_total, torch.zeros_like(class_tp))
    inv_freq = torch.where(class_total > 0, 1.0 / class_total, torch.zeros_like(class_total))
    inv_freq_sum = inv_freq.sum()
    if inv_freq_sum > 0:
        class_weights = inv_freq / inv_freq_sum
    else:
        class_weights = torch.full((num_classes,), 1.0 / num_classes)
    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2.0 * prec * recall / (prec + recall) if (prec > 0 and recall > 0) else torch.tensor(0.0)
    val_loss = loss / size if size > 0 else 0.0
    total_pos = class_total.sum()
    accuracy = class_tp.sum() / total_pos if total_pos > 0 else torch.tensor(0.0)
    results = {
        METRIC_LOSS: val_loss,
        METRIC_ACCURACY: accuracy,
        METRIC_F1: f1,
        METRIC_PRECISION: prec,
        METRIC_RECALL: recall,
    }

    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1 = 0 if (p <= 0 or r <= 0) else 2.0 * p * r / (p + r)
            class_results.append({METRIC_F1: f1, METRIC_PRECISION: p, METRIC_RECALL: r})
        results = results, class_results

    return results


def tune_threshold_on_validation(
    val_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    num_classes: int,
    device: torch.device,
    one_hot: bool = True,
    threshold_candidates: tuple[float, ...] | None = None,
    progress_label: str | None = None,
    apply_sigmoid: bool = True,
) -> tuple[float, dict[str, torch.Tensor | float]]:
    if threshold_candidates is None:
        threshold_candidates = tuple(i / 100 for i in range(5, 96, 5))

    net.eval()
    total_loss = 0.0
    size = len(val_loader.dataset)
    scores_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    progress_bar = None
    total_batches = len(val_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label=progress_label)

    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            logits = net(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            scores = torch.sigmoid(logits) if apply_sigmoid else logits
            if not one_hot:
                labels_mat = torch.zeros_like(scores)
                labels_mat[torch.arange(scores.shape[0]), labels] = 1.0
                labels = labels_mat
            scores_list.append(scores.cpu())
            labels_list.append(labels.cpu())
            if progress_bar:
                progress_bar.increment()

    if progress_bar:
        progress_bar.finish()

    all_scores = torch.cat(scores_list, dim=0) if scores_list else torch.zeros((0, num_classes))
    all_labels = torch.cat(labels_list, dim=0) if labels_list else torch.zeros((0, num_classes))
    val_loss = total_loss / size if size > 0 else 0.0

    best_threshold = threshold_candidates[0]
    best_results: dict[str, torch.Tensor | float] | None = None
    for threshold in threshold_candidates:
        predictions = torch.where(all_scores > threshold, 1.0, 0.0)
        metrics = _compute_weighted_multilabel_metrics(predictions, all_labels, num_classes)
        result = {METRIC_LOSS: val_loss, **metrics}
        if best_results is None or float(result[METRIC_F1]) > float(best_results[METRIC_F1]):
            best_threshold = threshold
            best_results = result

    if best_results is None:
        best_results = {
            METRIC_LOSS: val_loss,
            METRIC_ACCURACY: 0.0,
            METRIC_F1: 0.0,
            METRIC_PRECISION: 0.0,
            METRIC_RECALL: 0.0,
        }
    return float(best_threshold), best_results


class ProgressBarElements:
    PROGRESS_RATIO = "{progress_ratio}"
    PROGRESS_BAR = "{progress_bar}"
    PROGRESS_PERCENTAGE = "{progress_percentage}"
    LABEL = "{label}"


class ProgressBar:
    _DEFAULT_LAYOUT = (
        ProgressBarElements.PROGRESS_RATIO,
        " |",
        ProgressBarElements.PROGRESS_BAR,
        "| ",
        ProgressBarElements.PROGRESS_PERCENTAGE,
    )

    def __init__(
        self,
        total: int,
        start_at: int = 1,
        decimals: int = 1,
        length: int = 50,
        void: str = " ",
        fill: str = "#",
        print_end: str = "\r",
        layout: list[str] = None,
        label: str | None = None,
    ) -> None:
        self.iteration = start_at
        self.total = total
        self.decimals = decimals
        self.length = length
        self.void = void
        self.fill = fill
        self.print_end = print_end
        self._finished = False
        self.label = label or ""

        if layout is not None:
            self.layout = layout
        elif self.label:
            self.layout = [ProgressBarElements.LABEL, " ", *self._DEFAULT_LAYOUT]
        else:
            self.layout = list(self._DEFAULT_LAYOUT)

    def start(self):
        self.update()

    def update(self):
        if self.iteration > self.total:
            if not self._finished:
                self.finish()
            self._finished = True
            return

        try:
            self.percent = f"{self.iteration / self.total * 100: .{self.decimals}f}"
        except ZeroDivisionError:
            raise ValueError("Cannot have total = 0")

        filled_length = int(self.length * self.iteration // self.total)

        bar = self.fill * filled_length + self.void * (self.length - filled_length)

        full_bar = "".join(self.layout).format_map(
            {
                "progress_ratio": f"{self.iteration}/{self.total}",
                "progress_bar": bar,
                "progress_percentage": f"{self.percent}%",
                "label": self.label,
            }
        )

        progress_bar = f"{full_bar: <{os.get_terminal_size().columns}}"

        print(f"\r{progress_bar}", end=self.print_end)

    def increment(self):
        self.iteration += 1
        self.update()

    def clear_line(self):
        print("\r" + " " * os.get_terminal_size().columns, end="\r")

    def finish(self):
        print()
