import copy
import logging
import nibabel as nib
import numpy as np
import os
from omegaconf import DictConfig

import torch

from catalyst.utils.torch import load_checkpoint, unpack_checkpoint

from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.task.logreg_evaluation import LogRegEvaluationTask, \
    LogRegEvaluationTaskBuilder


class SaliencyTaskBuilder(LogRegEvaluationTaskBuilder):
    def create_new_task(self, task_args: DictConfig, seed: int = 343):
        self._task = SaliencyTask(task_args.args, seed=seed)

    def add_model(self, model_config: DictConfig):
        self._task.model = {}
        # get number of classes
        num_classes = self._task.dataset._num_classes
        if "num_classes" in model_config.args.keys():
            model_config.args["num_classes"] = num_classes
        pretrained_checkpoint = model_config.args.pretrained_checkpoint
        # create model
        model_args = copy.deepcopy({**model_config.args})
        model_args.pop("pretrained_checkpoint")
        pretrained_model = model_provider.get(model_config.name, **model_args)
        # load checkpoint
        checkpoint = load_checkpoint(pretrained_checkpoint)
        unpack_checkpoint(checkpoint, pretrained_model)
        # create linear evaluators
        dim_l = model_args["architecture_params"]["dim_l"]
        for source_id, encoder in pretrained_model.get_encoder_list().items():
            if model_args['architecture'] == 'DcganAutoEncoder':
                encoder = encoder._encoder
            smoothgrad_args = {
                "encoder": encoder,
                "source_id": int(source_id),
                "dim_l": dim_l,
            }
            print(smoothgrad_args)
            smoothgrad = model_provider.get(
                "SmoothGrad", **smoothgrad_args
            )
            self._task.model[source_id] = smoothgrad

    def add_runner(self, runner_config: DictConfig):
        pass


class SaliencyTask(LogRegEvaluationTask):
    def run(self):
        self._mask_mni = ~torch.BoolTensor(
            nib.load(self._dataset._template).get_fdata())
        sources = self._model.keys()
        for source_id in sources:
            self._reset_seed()
            logdir = self._task_args["logdir"] + f"/saliency_{source_id}/subjects/"
            index = 0
            for set_name in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
                for batch in self._dataset.get_loader(set_name):
                    logging.info("Computing...")
                    batch_pathes = batch[f'source_{source_id}']['path']
                    batch_saliency = self._model[source_id].forward(batch)
                    batch_affine = batch[f'source_{source_id}']['affine']
                    batch_images = batch[f'source_{source_id}']['data']
                    logging.info("Saving...")
                    for i in range(batch_images.size(0)):
                        self._save_npy_to_nifti(
                            logdir,
                            source_id,
                            batch_images[i],
                            batch_saliency[i],
                            batch_pathes[i],
                            batch_affine[i],
                            index
                        )
                        index += 1

    def _save_npy_to_nifti(
        self, logdir, source_id,
        image, saliency, base_nifti_filename,
        affine, index
    ):

        def generate_saliency_filename(
            logdir,
            file_type,
            subject_id,
            modifier='',
            file_extenstion='.nii.gz'
        ):
            path = (
                f'{logdir}/'
            )
            filename = f'{file_type}_{subject_id}{modifier}{file_extenstion}'
            # create path
            if not os.path.exists(path):
                os.makedirs(path)
            # combine path with filename
            filename = os.path.join(path, filename)
            return filename

        affine = np.eye(4)
        image = image.squeeze(0)
        # save image
        image = image.cpu().numpy()
        img = nib.Nifti1Image(image, affine)
        filename = generate_saliency_filename(
            logdir + f'/{index}/images/',
            'image', index
        )
        nib.save(img, filename)
        # save saliency
        logging.info(f'{saliency.max()}, {saliency.min()}')
        s_data = saliency.numpy().astype('float32')
        for i in range(s_data.shape[0]):
            t = s_data[i, :, :, :]
            t[self._mask_mni] = 0
            img = nib.Nifti1Image(t, affine)
            filename = generate_saliency_filename(
                logdir + f'{index}/saliencies/',
                'saliency',
                index,
                modifier=f'_{i}'
            )
            nib.save(img, filename)
            logging.debug(f'Saved image {filename}')
