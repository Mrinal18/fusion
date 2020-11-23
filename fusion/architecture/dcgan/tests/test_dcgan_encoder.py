from fusion.architecture.dcgan import DcganEncoder
import torch
import unittest


class TestDcganEncoder(unittest.TestCase):

    def test_forward(self):
        input_size = 32
        dim_in = 1
        dim_h = 2
        dim_l = 4
        dim_cls = [8]
        batch_size = 2
        encoder = DcganEncoder(dim_in, dim_h, dim_l, dim_cls, input_size=input_size)
        x = torch.rand(batch_size, dim_in, input_size, input_size)
        output = encoder(x)
        self.assertEqual(len(output), 2)
        z, latents = output
        self.assertEqual(z.size(0), batch_size)
        self.assertEqual(z.size(1), dim_l)
        for i, (d, l) in enumerate(latents.items()):
            if d != 1:
                self.assertEqual(l.size(0), batch_size)
                self.assertEqual(l.size(-1), dim_cls[i])
                self.assertEqual(len(l.size()), len(x.size()))


if __name__ == '__main__':
    unittest.main()
