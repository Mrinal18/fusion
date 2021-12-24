import nibabel as nib
from numpy import float64

from fusion.dataset.oasis.oasis import Oasis
from fusion.dataset.misc import SetId
import unittest


class TestOasis(unittest.TestCase):
    #@unittest.skip("Skipping Oasis, as it requires OASIS which is not open_sourced")
    def test_oasis(self):
        BATCH_SIZE = 512
        dataset_dir = "../../../../data/oasis/"
        template = "../../../../data/MNI152_T1_3mm_brain_mask_dil_cubic192.nii.gz"
        dataset = Oasis(
            dataset_dir=dataset_dir,
            template=template,
            sources=[0, 1],
            batch_size=BATCH_SIZE,
            only_labeled=True
        )
        dataset.load()
        #self.assertEqual(dataset.num_classes, 2)
        #self.assertEqual(len(dataset.get_loader(SetId.TRAIN)), 321)
        #self.assertEqual(len(dataset.get_loader(SetId.VALID)), 82)
        #self.assertEqual(len(dataset.get_loader(SetId.INFER)), 54)
        #self.assertEqual(len(dataset.get_cv_loaders()), 2)
        #self.assertEqual(len(dataset.get_all_loaders()), 3)

        import numpy as np
        import nibabel as nib
        t1 = None
        falff = None
        for _, loader in dataset.get_all_loaders().items():
            for batch in loader:
                x = [
                     v['data'] for k, v in batch.items() if k.startswith('source')
                ]
                batch_t1 = x[0].numpy().astype(float64)
                batch_falff = x[1].numpy().astype(float64)
                t1 = np.concatenate(
                    (t1, batch_t1), axis=0) if t1 is not None else batch_t1
                falff = np.concatenate(
                    (falff, batch_falff), axis=0) if falff is not None else batch_falff
        mean_t1_image = np.mean(t1[:2, :, :, :, :], axis=0)[0]
        #print (mean_t1_image.shape)
        mean_falff_image = np.mean(falff[:2, :, :, :, :], axis=0)[0]
        #print (mean_falff_image.shape)
        print (t1.shape)
        print (falff.shape)
        mean_t1_image = nib.Nifti1Image(mean_t1_image, np.eye(4))
        print (mean_t1_image.shape)
        mean_falff_image = nib.Nifti1Image(mean_falff_image, np.eye(4))
        print (mean_falff_image.shape)
        nib.save(mean_t1_image, 't1_mean_two.nii')
        nib.save(mean_falff_image, 'falff_mean_two.nii')

        TEMPLATE_FILENAME = '../../../../data/MNI152_T1_3mm_brain_cubic192.nii.gz'
        TEMPLATE = nib.load(TEMPLATE_FILENAME)
        TEMPLATE_AFFINE = TEMPLATE.affine
        TEMPLATE_HEADER = TEMPLATE.header
        #print (TEMPLATE_HEADER)
        print (nib.aff2axcodes(TEMPLATE.affine))

        canonical_img = nib.as_closest_canonical(TEMPLATE)
        print (nib.aff2axcodes(canonical_img.affine))


if __name__ == "__main__":
    unittest.main()
