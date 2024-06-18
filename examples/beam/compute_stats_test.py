from absl.testing import absltest
import apache_beam as beam
from firebench.examples.beam import compute_stats
import numpy as np
import xarray


class ComputeStatsXarrayTest(absltest.TestCase):

  def test_compute_stats(self):
    src_zarr = self.create_tempdir('source').full_path
    dest_zarr = self.create_tempdir('destination').full_path

    rs = np.random.RandomState(0)
    raw_data = rs.normal(size=(5, 100, 120, 20))
    dataset = xarray.Dataset({
        'u': (('t', 'x', 'y', 'z'), raw_data + 1),
        'v': (('t', 'x', 'y', 'z'), raw_data),
        'u0': (('t', 'x', 'y'), raw_data[..., 0] - 1),
    })
    source_chunks = {'t': 1, 'x': 50, 'y': 60, 'z': 20}
    dataset.chunk(source_chunks).to_zarr(src_zarr, consolidated=True)

    with beam.Pipeline('DirectRunner') as pipeline:
      _ = pipeline | compute_stats.ComputeStats(
          input_path=src_zarr, output_path=dest_zarr
      )

    actual = xarray.open_zarr(dest_zarr)

    # Check the new sizes.
    self.assertDictEqual(dict(actual.sizes), {'stat': 3, 't': 5})
    # Check the new chunk sizes.
    self.assertDictEqual(dict(actual.chunks), {'stat': (3,), 't': (5,)})
    # Check that the statistics are computed correctly.
    np.testing.assert_allclose(
        actual.u,
        [
            np.mean(raw_data + 1, axis=(1, 2, 3)),
            np.min(raw_data + 1, axis=(1, 2, 3)),
            np.max(raw_data + 1, axis=(1, 2, 3)),
        ],
    )
    np.testing.assert_allclose(
        actual.v,
        [
            np.mean(raw_data, axis=(1, 2, 3)),
            np.min(raw_data, axis=(1, 2, 3)),
            np.max(raw_data, axis=(1, 2, 3)),
        ],
    )
    np.testing.assert_allclose(
        actual.u0,
        [
            np.mean(raw_data[..., 0] - 1, axis=(1, 2)),
            np.min(raw_data[..., 0] - 1, axis=(1, 2)),
            np.max(raw_data[..., 0] - 1, axis=(1, 2)),
        ],
    )


if __name__ == '__main__':
  absltest.main()
