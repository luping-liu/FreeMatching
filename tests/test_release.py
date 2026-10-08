import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file
from freematching.geometry import warp_source, correspondence_to_flow
from freematching.checkpoint import load_checkpoint


class ReleaseTests(unittest.TestCase):
    def test_identity_and_one_pixel_translation(self):
        y, x = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 6), indexing='ij')
        grid = torch.stack([x, y])[None]
        source = torch.arange(24).float().reshape(1, 1, 4, 6)
        torch.testing.assert_close(warp_source(source, grid), source)
        torch.testing.assert_close(correspondence_to_flow(grid), torch.zeros_like(grid), atol=1e-6, rtol=0)
        shifted = grid.clone(); shifted[:, 0] += 2 / 5
        result = warp_source(source, shifted)
        torch.testing.assert_close(result[..., :-1], source[..., 1:])
        torch.testing.assert_close(correspondence_to_flow(shifted)[:, 0], torch.ones(1, 4, 6), atol=1e-6, rtol=0)

    def test_checkpoint_groups_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.safetensors'
            save_file({'vis.weight':torch.ones(2, 3), 'flow_decoder.bias':torch.ones(3)}, str(path))
            ckpt = load_checkpoint(path)
            self.assertEqual(ckpt['output_decoder'], 'flow')
            torch.testing.assert_close(ckpt['vis']['weight'], torch.ones(2, 3))
            save_file({'unexpected.weight':torch.ones(2)}, str(path))
            with self.assertRaises(ValueError): load_checkpoint(path)

    def test_video_tracks_roundtrip(self):
        import tifffile
        file = Path(__file__).resolve().parents[1] / 'scripts/prepare_data/export_video_tracks.py'
        spec = importlib.util.spec_from_file_location('export_video_tracks', file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        tracks = np.linspace(-1, 20, 5 * 2 * 3 * 4, dtype=np.float32).reshape(5, 2, 3, 4)
        conf = np.ones((5, 1, 3, 4), dtype=np.float32) * .8
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.savez(path/'tracks.npz',forward_tracks=tracks,backward_tracks=tracks,
                     forward_confidence=conf,backward_confidence=conf)
            module.export_tracks(path/'tracks.npz',path/'supp')
            for axis in range(2):
                q=tifffile.imread(path/'supp/fwd-003.tiff',key=axis)
                restored=tracks.min()+(q.astype(np.float32)/32767+1)*.5*(tracks.max()-tracks.min())
                np.testing.assert_allclose(restored,tracks[3,axis],atol=.0004)
            np.testing.assert_array_equal(tifffile.imread(path/'supp/bwd-003.tiff',key=2),204)

    def test_custom_transformer_dino_forward(self):
        from freematching.vendor.model_flux2 import Flux2Transformer2DModel
        model = Flux2Transformer2DModel(in_channels=8,out_channels=8,num_layers=1,num_single_layers=1,
            attention_head_dim=16,num_attention_heads=2,joint_attention_dim=32,
            timestep_guidance_channels=16,axes_dims_rope=(4,4,4,4),guidance_embeds=False)
        model.dino_proj = torch.nn.Linear(12,32)
        inputs=dict(hidden_states=torch.randn(1,8,8),timestep=torch.ones(1)*.62,
            encoder_hidden_states=torch.randn(1,2,32),txt_ids=torch.zeros(2,4),img_ids=torch.zeros(8,4),
            dino_repre=[torch.randn(1,4,12),torch.randn(1,4,12)])
        with torch.no_grad(): result = model(**inputs).sample
        self.assertEqual(result.shape,(1,8,8));self.assertTrue(torch.isfinite(result).all())
        inputs['dino_repre']=[torch.randn(1,3,12),torch.randn(1,4,12)]
        with self.assertRaises(ValueError): model(**inputs)


if __name__ == '__main__':
    unittest.main()
