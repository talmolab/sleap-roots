"""Performance benchmarks for trait extraction pipelines.

This module benchmarks the end-to-end trait extraction time for all pipeline classes.
Benchmarks use real test data and measure pipeline performance.

Hardware: GitHub Actions Ubuntu 22.04 runners (2 cores, 7GB RAM)
Expected performance (per plant):
- Single plant pipelines: ~0.1-0.5s
- Multiple plant pipelines: ~0.5-2s (varies with plant count)
"""

import sleap_roots as sr


class TestSinglePlantPipelines:
    """Benchmarks for single-plant pipeline performance."""

    def test_dicot_pipeline_performance(
        self,
        benchmark,
        canola_h5,
        canola_primary_slp,
        canola_lateral_slp,
    ):
        """Benchmark DicotPipeline.compute_plant_traits() on canola plant.

        Dataset: Single canola plant (7 DAG) with primary + lateral roots
        Expected: ~0.1-0.5s per plant on GitHub Actions runners
        """
        series = sr.Series.load(
            "canola_7do",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
            lateral_path=canola_lateral_slp,
        )
        pipeline = sr.DicotPipeline()

        # Benchmark the trait extraction
        result = benchmark(pipeline.compute_plant_traits, series)

        # Sanity check: ensure traits were computed
        assert result is not None
        assert len(result) > 0

    def test_younger_monocot_pipeline_performance(
        self,
        benchmark,
        rice_h5,
        rice_long_slp,
        rice_main_slp,
    ):
        """Benchmark YoungerMonocotPipeline.compute_plant_traits() on young rice.

        Dataset: Rice plant (3 DAG) with primary + crown roots
        Expected: ~0.1-0.5s per plant on GitHub Actions runners
        """
        series = sr.Series.load(
            "rice_3do",
            h5_path=rice_h5,
            primary_path=rice_long_slp,
            crown_path=rice_main_slp,
        )
        pipeline = sr.YoungerMonocotPipeline()

        result = benchmark(pipeline.compute_plant_traits, series)

        assert result is not None
        assert len(result) > 0

    def test_older_monocot_pipeline_performance(
        self,
        benchmark,
        rice_main_10do_h5,
        rice_main_10do_slp,
    ):
        """Benchmark OlderMonocotPipeline.compute_plant_traits() on mature rice.

        Dataset: Rice plant (10 DAG) with crown roots only
        Expected: ~0.1-0.5s per plant on GitHub Actions runners
        """
        series = sr.Series.load(
            "rice_10do",
            h5_path=rice_main_10do_h5,
            crown_path=rice_main_10do_slp,
        )
        pipeline = sr.OlderMonocotPipeline()

        result = benchmark(pipeline.compute_plant_traits, series)

        assert result is not None
        assert len(result) > 0

    def test_primary_root_pipeline_performance(
        self,
        benchmark,
        canola_h5,
        canola_primary_slp,
    ):
        """Benchmark PrimaryRootPipeline.compute_plant_traits() on primary root.

        Dataset: Single canola plant with primary root only
        Expected: ~0.1-0.3s per plant (faster than full dicot pipeline)
        """
        series = sr.Series.load(
            "canola_primary_only",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )
        pipeline = sr.PrimaryRootPipeline()

        result = benchmark(pipeline.compute_plant_traits, series)

        assert result is not None
        assert len(result) > 0

    def test_lateral_root_pipeline_performance(
        self,
        benchmark,
        canola_h5,
        canola_lateral_slp,
    ):
        """Benchmark LateralRootPipeline.compute_plant_traits() on lateral roots.

        Dataset: Single canola plant with lateral roots only
        Expected: ~0.1-0.3s per plant
        """
        series = sr.Series.load(
            "canola_lateral_only",
            h5_path=canola_h5,
            lateral_path=canola_lateral_slp,
        )
        pipeline = sr.LateralRootPipeline()

        result = benchmark(pipeline.compute_plant_traits, series)

        assert result is not None
        assert len(result) > 0


class TestMultiplePlantPipelines:
    """Benchmarks for multi-plant batch processing pipelines."""

    def test_multiple_dicot_pipeline_performance(
        self,
        benchmark,
        canola_h5,
        canola_primary_slp,
        canola_lateral_slp,
    ):
        """Benchmark MultipleDicotPipeline.compute_multiple_dicots_traits().

        Dataset: Canola plant data (may contain multiple plants)
        Expected: ~0.5-2s depending on plant count
        Note: Performance scales with number of plants detected
        """
        series = sr.Series.load(
            "canola_multi",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
            lateral_path=canola_lateral_slp,
        )
        pipeline = sr.MultipleDicotPipeline()

        result = benchmark(pipeline.compute_multiple_dicots_traits, series)

        assert result is not None
        assert len(result) > 0

    def test_multiple_primary_root_pipeline_performance(
        self,
        benchmark,
        canola_h5,
        canola_primary_slp,
    ):
        """Benchmark MultiplePrimaryRootPipeline.compute_multiple_primary_roots_traits().

        Dataset: Canola with primary roots
        Expected: ~0.5-2s depending on plant count
        """
        series = sr.Series.load(
            "canola_multi_primary",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )
        pipeline = sr.MultiplePrimaryRootPipeline()

        result = benchmark(pipeline.compute_multiple_primary_roots_traits, series)

        assert result is not None
        assert len(result) > 0
