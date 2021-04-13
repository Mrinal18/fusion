from fusion.architecture.dcgan.vae_encoder import VAEEncoder
import torch
import unittest


class TestVAEEncoder(unittest.TestCase):

    def test_forward(self):
        # define parameters
        input_size = 32
        dim_in = 1
        dim_h = 2
        dim_l = 4
        dim_cls = [8]
        batch_size = 2
        # create encoder
        encoder = VAEEncoder(dim_in, dim_h, dim_l, dim_cls, 
                             input_size=input_size)
        # create input
        x = torch.rand(batch_size, dim_in, input_size, input_size)
        # forward pass
        output = encoder(x)
        # check outputs
        self.assertEqual(len(output), 2)
        (mean, logvar), latents = output
        self.assertEqual(mean.size(0), batch_size)
        self.assertEqual(mean.size(1), dim_l)
        self.assertEqual(logvar.size(0), batch_size)
        self.assertEqual(logvar.size(1), dim_l)
        for i, (d, l) in enumerate(latents.items()):
            if d != 1:
                self.assertEqual(l.size(0), batch_size)
                self.assertEqual(l.size(-1), dim_cls[i])
                self.assertEqual(len(l.size()), len(x.size()))


if __name__ == '__main__':
    unittest.main()