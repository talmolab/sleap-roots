# Batch Processing Optimization

This recipe shows how to efficiently process large numbers of plants and optimize performance for high-throughput phenotyping.

## Problem

Processing hundreds or thousands of plants can be slow and memory-intensive. You need to:

- Process large datasets efficiently
- Minimize memory usage
- Utilize multiple CPU cores
- Handle errors gracefully
- Track progress

## Solution Overview

Optimize batch processing through:
1. **Parallel processing**: Use multiprocessing
2. **Memory management**: Process in batches, clear unused data
3. **Progress tracking**: Monitor long-running jobs
4. **Error handling**: Continue despite failures
5. **Smart I/O**: Efficient file reading/writing

## Basic Batch Processing

### Sequential Processing

```python
import sleap_roots as sr
from pathlib import Path

# Find all H5 files
data_dir = Path("data/")
h5_files = list(data_dir.glob("*.h5"))

print(f"Found {len(h5_files)} plants to process")

# Process each plant
pipeline = sr.DicotPipeline()
all_traits = []

for h5_file in h5_files:
    print(f"Processing {h5_file.name}...")

    series = sr.Series.load(
        series_name=h5_file.stem,
        h5_path=h5_file,
        primary_path=h5_file.with_suffix(".primary.slp"),
        lateral_path=h5_file.with_suffix(".lateral.slp")
    )

    traits = pipeline.compute_plant_traits(series)
    traits['plant_id'] = h5_file.stem
    all_traits.append(traits)

# Combine results
import pandas as pd
combined_traits = pd.concat(all_traits, ignore_index=True)
combined_traits.to_csv("all_traits.csv", index=False)

print(f"Processed {len(all_traits)} plants")
```

## Parallel Processing

### Using multiprocessing

```python
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_plant(h5_file, primary_suffix=".primary.slp", lateral_suffix=".lateral.slp"):
    """
    Process a single plant (function for parallel execution).

    Args:
        h5_file: Path to H5 file
        primary_suffix: Suffix for primary root file
        lateral_suffix: Suffix for lateral root file

    Returns:
        DataFrame with traits, or None if processing failed
    """
    try:
        series = sr.Series.load(
            series_name=h5_file.stem,
            h5_path=h5_file,
            primary_path=h5_file.with_suffix(primary_suffix),
            lateral_path=h5_file.with_suffix(lateral_suffix)
        )

        pipeline = sr.DicotPipeline()
        traits = pipeline.compute_plant_traits(series)
        traits['plant_id'] = h5_file.stem

        return traits

    except Exception as e:
        print(f"Error processing {h5_file.name}: {e}")
        return None

# Parallel processing
h5_files = list(Path("data/").glob("*.h5"))

# Use 75% of available cores
n_workers = max(1, int(cpu_count() * 0.75))
print(f"Processing {len(h5_files)} plants with {n_workers} workers...")

with Pool(processes=n_workers) as pool:
    results = pool.map(process_single_plant, h5_files)

# Filter out failed plants and combine
successful_results = [r for r in results if r is not None]
combined_traits = pd.concat(successful_results, ignore_index=True)
combined_traits.to_csv("all_traits.csv", index=False)

print(f"Successfully processed {len(successful_results)}/{len(h5_files)} plants")
```

### With Progress Bar

```python
from tqdm import tqdm
from multiprocessing import Pool

# Process with progress bar
h5_files = list(Path("data/").glob("*.h5"))
n_workers = 4

with Pool(processes=n_workers) as pool:
    results = list(tqdm(
        pool.imap(process_single_plant, h5_files),
        total=len(h5_files),
        desc="Processing plants"
    ))

successful = [r for r in results if r is not None]
print(f"\nSuccessfully processed: {len(successful)}/{len(h5_files)}")
```

## Memory Management

### Batch Processing

Process in chunks to limit memory usage:

```python
def process_in_batches(h5_files, batch_size=100, output_dir="output"):
    """
    Process files in batches to manage memory.

    Args:
        h5_files: List of H5 file paths
        batch_size: Number of files per batch
        output_dir: Directory for batch output files

    Returns:
        List of batch output file paths
    """
    from pathlib import Path
    import gc

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    batch_files = []
    pipeline = sr.DicotPipeline()

    for batch_idx in range(0, len(h5_files), batch_size):
        batch = h5_files[batch_idx:batch_idx + batch_size]

        print(f"Processing batch {batch_idx//batch_size + 1}/{len(h5_files)//batch_size + 1}...")

        batch_results = []
        for h5_file in tqdm(batch, desc="Batch progress"):
            try:
                series = sr.Series.load(
                    series_name=h5_file.stem,
                    h5_path=h5_file,
                    primary_path=h5_file.with_suffix(".primary.slp"),
                    lateral_path=h5_file.with_suffix(".lateral.slp")
                )

                traits = pipeline.compute_plant_traits(series)
                traits['plant_id'] = h5_file.stem
                batch_results.append(traits)

                # Clear series to free memory
                del series

            except Exception as e:
                print(f"Error: {h5_file.name} - {e}")

        # Save batch results
        if batch_results:
            batch_df = pd.concat(batch_results, ignore_index=True)
            batch_file = output_dir / f"batch_{batch_idx//batch_size:03d}.csv"
            batch_df.to_csv(batch_file, index=False)
            batch_files.append(batch_file)

            # Clear batch results from memory
            del batch_results
            del batch_df
            gc.collect()

    return batch_files

# Process large dataset in batches
h5_files = list(Path("data/").glob("*.h5"))
batch_files = process_in_batches(h5_files, batch_size=100)

# Combine batch files
all_dfs = [pd.read_csv(f) for f in batch_files]
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv("all_traits_combined.csv", index=False)
```

### Memory-Efficient Loading

```python
def process_with_memory_limit(h5_files, max_memory_mb=2000):
    """
    Process plants while monitoring memory usage.

    Args:
        h5_files: List of H5 files
        max_memory_mb: Maximum memory to use (MB)

    Returns:
        DataFrame with results
    """
    import psutil
    import gc

    process = psutil.Process()
    results = []
    pipeline = sr.DicotPipeline()

    for h5_file in h5_files:
        # Check memory before loading
        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb > max_memory_mb:
            print(f"Memory limit reached ({memory_mb:.0f}MB), forcing garbage collection...")
            gc.collect()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > max_memory_mb:
                print(f"Still over limit, saving intermediate results...")
                # Save and clear results
                if results:
                    intermediate_df = pd.concat(results)
                    intermediate_df.to_csv(
                        f"intermediate_{len(results)}.csv",
                        index=False
                    )
                    results = []
                    gc.collect()

        # Process plant
        try:
            series = sr.Series.load(
                series_name=h5_file.stem,
                h5_path=h5_file,
                primary_path=h5_file.with_suffix(".primary.slp"),
                lateral_path=h5_file.with_suffix(".lateral.slp")
            )
            traits = pipeline.compute_plant_traits(series)
            traits['plant_id'] = h5_file.stem
            results.append(traits)

            del series
        except Exception as e:
            print(f"Error: {h5_file.name} - {e}")

    return pd.concat(results) if results else pd.DataFrame()
```

## Error Handling

### Robust Processing

```python
import logging
from datetime import datetime

def setup_logging(log_file="processing.log"):
    """Set up logging for batch processing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def process_plant_robust(h5_file, max_retries=3):
    """
    Process plant with error handling and retries.

    Args:
        h5_file: Path to H5 file
        max_retries: Number of retry attempts

    Returns:
        Tuple of (traits DataFrame, status dict)
    """
    status = {
        'plant_id': h5_file.stem,
        'success': False,
        'error': None,
        'retries': 0,
        'processing_time': 0
    }

    start_time = datetime.now()

    for attempt in range(max_retries):
        try:
            series = sr.Series.load(
                series_name=h5_file.stem,
                h5_path=h5_file,
                primary_path=h5_file.with_suffix(".primary.slp"),
                lateral_path=h5_file.with_suffix(".lateral.slp")
            )

            pipeline = sr.DicotPipeline()
            traits = pipeline.compute_plant_traits(series)
            traits['plant_id'] = h5_file.stem

            status['success'] = True
            status['processing_time'] = (datetime.now() - start_time).total_seconds()

            logging.info(f"Successfully processed {h5_file.stem}")
            return traits, status

        except Exception as e:
            status['retries'] = attempt + 1
            status['error'] = str(e)
            logging.warning(f"Attempt {attempt+1}/{max_retries} failed for {h5_file.stem}: {e}")

            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
            else:
                logging.error(f"All retries failed for {h5_file.stem}")
                return None, status

    return None, status

# Batch process with error tracking
setup_logging()
h5_files = list(Path("data/").glob("*.h5"))

all_traits = []
all_statuses = []

for h5_file in tqdm(h5_files):
    traits, status = process_plant_robust(h5_file)

    if traits is not None:
        all_traits.append(traits)

    all_statuses.append(status)

# Save results and status report
if all_traits:
    combined_traits = pd.concat(all_traits, ignore_index=True)
    combined_traits.to_csv("all_traits.csv", index=False)

status_df = pd.DataFrame(all_statuses)
status_df.to_csv("processing_status.csv", index=False)

# Print summary
print(f"\nProcessing Summary:")
print(f"Total plants: {len(all_statuses)}")
print(f"Successful: {status_df['success'].sum()}")
print(f"Failed: {(~status_df['success']).sum()}")
print(f"Mean processing time: {status_df[status_df['success']]['processing_time'].mean():.2f}s")
```

## Progress Tracking

### Advanced Progress Monitoring

```python
from tqdm import tqdm
import time

class ProgressTracker:
    """Track processing progress with detailed metrics."""

    def __init__(self, total_plants):
        self.total = total_plants
        self.processed = 0
        self.failed = 0
        self.start_time = time.time()
        self.pbar = tqdm(total=total_plants, desc="Processing")

    def update(self, success=True):
        """Update progress."""
        self.processed += 1
        if not success:
            self.failed += 1

        # Update progress bar
        self.pbar.update(1)
        self.pbar.set_postfix({
            'success': self.processed - self.failed,
            'failed': self.failed,
            'rate': f'{self._get_rate():.1f}/s'
        })

    def _get_rate(self):
        """Get processing rate."""
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0

    def finish(self):
        """Finish progress tracking."""
        self.pbar.close()

        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed

        print(f"\nProcessing completed in {elapsed:.1f}s")
        print(f"Rate: {rate:.2f} plants/second")
        print(f"Success: {self.processed - self.failed}/{self.total}")

        if self.failed > 0:
            print(f"Failed: {self.failed} ({self.failed/self.total*100:.1f}%)")

# Usage
h5_files = list(Path("data/").glob("*.h5"))
tracker = ProgressTracker(len(h5_files))

for h5_file in h5_files:
    success = process_single_plant(h5_file) is not None
    tracker.update(success=success)

tracker.finish()
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_processing(h5_file):
    """Profile single plant processing."""

    profiler = cProfile.Profile()
    profiler.enable()

    # Process plant
    series = sr.Series.load(
        series_name=h5_file.stem,
        h5_path=h5_file,
        primary_path=h5_file.with_suffix(".primary.slp"),
        lateral_path=h5_file.with_suffix(".lateral.slp")
    )
    pipeline = sr.DicotPipeline()
    traits = pipeline.compute_plant_traits(series)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# Profile to find bottlenecks
h5_file = Path("data/sample.h5")
profile_processing(h5_file)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_series_cached(h5_path, primary_path, lateral_path):
    """
    Load series with caching (for repeated processing).

    Warning: Only use if processing same files multiple times.
    """
    return sr.Series.load(
        series_name=Path(h5_path).stem,
        h5_path=h5_path,
        primary_path=primary_path,
        lateral_path=lateral_path
    )
```

## Complete Batch Processing System

### Production-Ready Implementation

```python
import sleap_roots as sr
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import logging
import time
import gc

class BatchProcessor:
    """
    Production-ready batch processor for plant phenotyping.

    Features:
    - Parallel processing
    - Progress tracking
    - Error handling
    - Memory management
    - Logging
    """

    def __init__(
        self,
        pipeline_class=sr.DicotPipeline,
        n_workers=None,
        batch_size=100,
        output_dir="output",
        log_file="processing.log"
    ):
        """
        Initialize batch processor.

        Args:
            pipeline_class: Pipeline class to use
            n_workers: Number of parallel workers (None = auto)
            batch_size: Batch size for memory management
            output_dir: Output directory
            log_file: Log file path
        """
        self.pipeline_class = pipeline_class
        self.n_workers = n_workers or max(1, int(cpu_count() * 0.75))
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_directory(self, data_dir, pattern="*.h5"):
        """
        Process all files in directory.

        Args:
            data_dir: Directory containing data files
            pattern: Glob pattern for finding files

        Returns:
            DataFrame with all traits
        """
        data_dir = Path(data_dir)
        h5_files = sorted(data_dir.glob(pattern))

        self.logger.info(f"Found {len(h5_files)} files to process")
        self.logger.info(f"Using {self.n_workers} workers")

        # Process in batches
        all_results = []
        start_time = time.time()

        for batch_idx in range(0, len(h5_files), self.batch_size):
            batch = h5_files[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            total_batches = (len(h5_files) + self.batch_size - 1) // self.batch_size

            self.logger.info(f"Processing batch {batch_num}/{total_batches}")

            batch_results = self._process_batch(batch)
            all_results.extend(batch_results)

            # Force garbage collection after batch
            gc.collect()

        # Combine results
        successful = [r for r in all_results if r is not None]
        combined = pd.concat(successful, ignore_index=True) if successful else pd.DataFrame()

        # Save results
        output_file = self.output_dir / "all_traits.csv"
        combined.to_csv(output_file, index=False)

        # Log summary
        elapsed = time.time() - start_time
        self.logger.info(f"Processing completed in {elapsed:.1f}s")
        self.logger.info(f"Successfully processed {len(successful)}/{len(h5_files)} plants")
        self.logger.info(f"Results saved to {output_file}")

        return combined

    def _process_batch(self, h5_files):
        """Process a batch of files in parallel."""
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self._process_single, h5_files),
                total=len(h5_files),
                desc="Batch progress"
            ))
        return results

    def _process_single(self, h5_file):
        """Process single plant."""
        try:
            series = sr.Series.load(
                series_name=h5_file.stem,
                h5_path=h5_file,
                primary_path=h5_file.with_suffix(".primary.slp"),
                lateral_path=h5_file.with_suffix(".lateral.slp")
            )

            pipeline = self.pipeline_class()
            traits = pipeline.compute_plant_traits(series)
            traits['plant_id'] = h5_file.stem
            traits['h5_file'] = str(h5_file)

            return traits

        except Exception as e:
            self.logger.error(f"Error processing {h5_file.name}: {e}")
            return None

# Usage
processor = BatchProcessor(
    pipeline_class=sr.DicotPipeline,
    n_workers=4,
    batch_size=100,
    output_dir="results"
)

traits = processor.process_directory("data/", pattern="*.h5")
print(f"Processed {len(traits)} plants")
```

## Benchmarking

### Performance Comparison

```python
import time

def benchmark_processing_methods(h5_files):
    """Compare different processing methods."""

    results = {}

    # Sequential
    start = time.time()
    sequential_results = []
    for h5_file in h5_files[:10]:  # Test on subset
        result = process_single_plant(h5_file)
        sequential_results.append(result)
    results['sequential'] = time.time() - start

    # Parallel
    start = time.time()
    with Pool(processes=4) as pool:
        parallel_results = pool.map(process_single_plant, h5_files[:10])
    results['parallel_4workers'] = time.time() - start

    # Print comparison
    print("Benchmark Results (10 plants):")
    for method, elapsed in results.items():
        print(f"{method}: {elapsed:.2f}s ({10/elapsed:.2f} plants/s)")

# Run benchmark
h5_files = list(Path("data/").glob("*.h5"))
benchmark_processing_methods(h5_files)
```

## Next Steps

- See [Filtering Data](filtering-data.md) for data quality improvements
- Read [Batch Processing Guide](../guides/batch-processing.md) for general workflow
- Check [Troubleshooting](../guides/troubleshooting.md) for common issues