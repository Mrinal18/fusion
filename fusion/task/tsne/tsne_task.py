from catalyst.utils.torch import load_checkpoint, unpack_checkpoint
import copy
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from omegaconf import DictConfig
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.task.logreg_evaluation import LogRegEvaluationTask, \
    LogRegEvaluationTaskBuilder


class TsneTaskBuilder(LogRegEvaluationTaskBuilder):
    def create_new_task(self, task_args: DictConfig, seed: int = 343):
        """
        Method to create new TSNE task

        Args:
            task_args: dictionary with task's parameters from config
        """
        self._task = TsneTask(task_args.args, seed=seed)

    def add_model(self, model_config: DictConfig):
        """
        Method for add model to linear evaluation task
        Args:
                model_config: dictionary with model's parameters from config
        """
        self._task.model = {}
        # get number of classes
        #num_classes = self._task.dataset._num_classes
        if "num_classes" in model_config.args.keys():
            model_config.args["num_classes"] = 2
        pretrained_checkpoint = model_config.args.pretrained_checkpoint
        # create model
        model_args = copy.deepcopy({**model_config.args})
        model_args.pop("pretrained_checkpoint")
        pretrained_model = model_provider.get(model_config.name, **model_args)
        # load checkpoint
        checkpoint = load_checkpoint(pretrained_checkpoint)
        unpack_checkpoint(checkpoint, pretrained_model)
        # create linear evaluators
        for source_id, encoder in pretrained_model.get_encoder_list().items():
            encoder_extractor_args = {
                "encoder": encoder,
                "source_id": int(source_id),
            }
            print(encoder_extractor_args)
            encoder_extractor = model_provider.get(
                "EncoderExtractor", **encoder_extractor_args
            )
            self._task.model[source_id] = encoder_extractor


class TsneTask(LogRegEvaluationTask):
    def run(self):
        sns.set_style('whitegrid')
        for source_id in self._model.keys():
            tsne = TSNE(**self._task_args['tsne_args'])
            plt.figure(dpi=300)
            scaler = StandardScaler()
            self._reset_seed()
            logdir = self._task_args["logdir"] + f"/tsne_{source_id}/"
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            tsne_projections = None
            all_targets = None
            for set_id, set_name in enumerate([SetId.TRAIN, SetId.VALID, SetId.INFER]):
                representation, targets = self._get_representation(
                    source_id, set_name
                )
                if set_name == SetId.TRAIN:
                    scaler.fit(representation)
                scaled_representation = scaler.transform(representation)
                tsne_projections = self._concat_x_to_y(
                    scaled_representation, tsne_projections)
                all_targets = self._concat_x_to_y(
                    targets, all_targets
                )
            tsne_projections = tsne.fit_transform(tsne_projections)
            vis_x = tsne_projections[:, 0]
            vis_y = tsne_projections[:, 1]
            plt.scatter(vis_x, vis_y, c=all_targets,
                        label=all_targets, cmap=plt.cm.get_cmap("jet", 3), marker='.')
            plt.colorbar(ticks=range(3))
            plt.savefig(f'{logdir}/tsne.png', dpi=300, bbox_inches='tight')

