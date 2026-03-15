from utils.references import METRIC_ACCURACY, METRIC_F1, METRIC_LOSS, METRIC_PRECISION, METRIC_RECALL
from torch.utils.tensorboard.summary import hparams as build_hparams_summary


def configure_custom_scalar_layout(summary_writer, lr_group_count: int) -> None:
    layout = {
        "Overview": {
            "Loss": ["Multiline", ["Metrics/Loss/train", "Metrics/Loss/validation"]],
            "Accuracy": ["Multiline", ["Metrics/Accuracy/train", "Metrics/Accuracy/validation"]],
            "F1": ["Multiline", ["Metrics/F1/train", "Metrics/F1/validation"]],
            "Precision": ["Multiline", ["Metrics/Precision/train", "Metrics/Precision/validation"]],
            "Recall": ["Multiline", ["Metrics/Recall/train", "Metrics/Recall/validation"]],
        }
    }
    if lr_group_count > 0:
        layout["Overview"]["Learning Rate"] = [
            "Multiline",
            [f"LR/group_{group_index}" for group_index in range(lr_group_count)],
        ]
    summary_writer.add_custom_scalars(layout)


def log_run_configuration(summary_writer, run_name: str, run_config: dict[str, object]) -> None:
    summary_writer.add_text("Run/name", run_name, 0)
    table_lines = [
        "| Parameter | Value |",
        "| --- | --- |",
    ]
    for key in sorted(run_config):
        value = str(run_config[key]).replace("|", "\\|")
        table_lines.append(f"| {key} | {value} |")
    summary_writer.add_text("Run/configuration", "\n".join(table_lines), 0)


def log_hparams_summary(
    summary_writer, hparams: dict[str, bool | str | float | int], metrics: dict[str, float]
) -> None:
    exp, ssi, sei = build_hparams_summary(hparams, metrics)
    file_writer = summary_writer._get_file_writer()  # TensorBoard hparams plugin requires raw summaries.
    file_writer.add_summary(exp)
    file_writer.add_summary(ssi)
    file_writer.add_summary(sei)
    for metric_name, metric_value in metrics.items():
        summary_writer.add_scalar(metric_name, float(metric_value), 0)
    summary_writer.flush()


def log_metric_pair(summary_writer, tag_root, train_value, validation_value, step):
    summary_writer.add_scalar(f"{tag_root}/train", train_value, step)
    summary_writer.add_scalar(f"{tag_root}/validation", validation_value, step)


def update_graphs(
    summary_writer,
    epoch,
    train_results,
    validation_results,
    train_class_results=None,
    test_class_results=None,
    class_names=None,
    mbatch_group=-1,
    mbatch_count=0,
    mbatch_losses=None,
):
    step = (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count

    if mbatch_group > 0:
        for i in range(len(mbatch_losses)):
            summary_writer.add_scalar(
                "Losses/Train mini-batches", mbatch_losses[i], epoch * mbatch_count + (i + 1) * mbatch_group
            )

    log_metric_pair(
        summary_writer,
        "Metrics/Loss",
        train_results[METRIC_LOSS],
        validation_results[METRIC_LOSS],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/Accuracy",
        train_results[METRIC_ACCURACY],
        validation_results[METRIC_ACCURACY],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/F1",
        train_results[METRIC_F1],
        validation_results[METRIC_F1],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/Precision",
        train_results[METRIC_PRECISION],
        validation_results[METRIC_PRECISION],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/Recall",
        train_results[METRIC_RECALL],
        validation_results[METRIC_RECALL],
        step,
    )

    if train_class_results and test_class_results:
        for i in range(len(train_class_results)):
            summary_writer.add_scalar(
                f"ClassMetrics/{class_names[i]}/f1/train",
                train_class_results[i][METRIC_F1],
                step,
            )
            summary_writer.add_scalar(
                f"ClassMetrics/{class_names[i]}/f1/validation",
                test_class_results[i][METRIC_F1],
                step,
            )
            summary_writer.add_scalar(
                f"ClassMetrics/{class_names[i]}/precision/train",
                train_class_results[i][METRIC_PRECISION],
                step,
            )
            summary_writer.add_scalar(
                f"ClassMetrics/{class_names[i]}/precision/validation",
                test_class_results[i][METRIC_PRECISION],
                step,
            )
            summary_writer.add_scalar(
                f"ClassMetrics/{class_names[i]}/recall/train",
                train_class_results[i][METRIC_RECALL],
                step,
            )
            summary_writer.add_scalar(
                f"ClassMetrics/{class_names[i]}/recall/validation",
                test_class_results[i][METRIC_RECALL],
                step,
            )
    summary_writer.flush()
