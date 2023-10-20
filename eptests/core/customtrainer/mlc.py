"""
Just overwrites test script for mlc. Train script is the same
"""
from baseline.pytorch.classify.train import ClassifyTrainerPyTorch
import torch
import torch.autograd
from sklearn.metrics import f1_score

from eight_mile.progress import create_progress_bar
from baseline.train import register_trainer


def mlc_compute_metrics(metrics, all_pred, all_ys, thresh=0.5):
    all_ys_concat = torch.cat(all_ys, dim=0).cpu().detach().numpy()
    all_pred_concat = torch.cat(all_pred, dim=0).cpu().detach().numpy()
    all_pred_concat[all_pred_concat >= 0.5] = 1
    all_pred_concat[all_pred_concat < 0.5] = 0
    # print(all_ys_concat[0])
    micro_f1 = f1_score(y_true=all_ys_concat, y_pred=all_pred_concat, average="micro")
    macro_f1 = f1_score(y_true=all_ys_concat, y_pred=all_pred_concat, average="macro")
    metrics["micro_f1"] = micro_f1
    metrics["macro_f1"] = macro_f1
    # print(micro_f1,macro_f1)
    # assert math.floor(metrics['macro_f1'])==math.floor(f1_score(y_true=true_labels, y_pred=preds, average="macro"))
    return metrics


@register_trainer(task="classify", name="mlc")
class ClassifyMLCTrainerPyTorch(ClassifyTrainerPyTorch):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def _test(self, loader, **kwargs):
        self.model.eval()
        total_loss = 0
        total_norm = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        all_ys = []
        all_pred = []
        with torch.no_grad():
            for batch_dict in pg(loader):
                example = self._make_input(batch_dict)
                ys = example.pop("y")
                pred = self.model(example)
                loss = self.crit(pred, ys)
                pred = torch.sigmoid(pred)  # normalize scores.
                batchsz = self._get_batchsz(batch_dict)
                total_loss += loss.item() * batchsz
                total_norm += batchsz
                # _add_to_cm(cm, ys, pred)
                all_ys.append(ys)
                all_pred.append(pred)
        # metrics = cm.get_all_metrics() if cm is not None else {}
        metrics = {}
        metrics = mlc_compute_metrics(metrics, all_pred, all_ys)
        metrics["avg_loss"] = total_loss / float(total_norm)
        # verbose_output(verbose, cm)
        # if handle is not None:
        #     handle.close()

        return metrics
