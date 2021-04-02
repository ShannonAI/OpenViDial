import torch
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("mse-loss")
class MSELoss(FairseqCriterion):
    """
    Implementation for the MSELoss.
    """

    def __init__(self, task):
        super().__init__(task)
        self.loss = torch.nn.MSELoss(reduction='mean')
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample['nsentences']
        loss = model(**sample["net_input"])
        target = torch.ones(sample['net_input']['src_tokens'].shape[0]).cuda()
        loss = self.loss(loss, target)

        logging_output = {
            "loss": loss,
            "ntokens": sample_size,
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
