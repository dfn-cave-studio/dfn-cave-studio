# M7 Demo Dataset

Synthetic borehole dataset for testing the full M7 data pre-processing pipeline.

## Files

| File | Description | Rows |
|------|-------------|------|
| `collars.csv` | 8 borehole collars + 2 deliberate error rows | 10 |
| `surveys.csv` | Survey stations every 10m + 2 error rows | ~150 |
| `fractures.csv` | 80 valid fracture observations + 3 auditable error rows | 83 |
| `rqd.csv` | RQD intervals + 3 error rows (overlap, RQD>100, from>to) | 27 |
| `domain_intervals.csv` | Domain assignments per borehole | 10 |

## Deliberate Issues

| Issue | Location | Purpose |
|-------|----------|---------|
| Missing hole_id | collars.csv row 9 | Test required-field validation |
| Negative total_depth | collars.csv row 10 | Test range validation |
| Survey depth > total depth | surveys.csv | Test depth-exceed check |
| Duplicate survey depth | surveys.csv BH-01 | Test depth-order check |
| Fracture depth > total depth | fractures.csv BH-01 | Test depth-exceed check |
| Dip out of range (999°) | fractures.csv BH-02 | Test angle validation |
| Non-integer set_id | fractures.csv BH-03 | Test type validation |
| RQD interval overlap | rqd.csv BH-01 | Test overlap detection |
| RQD > 100 | rqd.csv BH-02 | Test range check |
| from_depth >= to_depth | rqd.csv BH-03 | Test interval ordering |
| Domain interval overlap | domain_intervals.csv | Test overlap detection |
| **359°/1° wrapping** | fractures.csv Set 3 | Test angular wrap in clustering |

## Structure

- **8 boreholes**: BH-01 through BH-08
- **2 structural domains**: Domain 1 (BH-01,02,03,08), Domain 2 (BH-04,05,06,07)
- **3 joint sets**:
  - Set 1: dd≈45°, dip≈60° (NE-SW sub-vertical)
  - Set 2: dd≈135°, dip≈75° (NW-SE sub-vertical)
  - Set 3: dd≈270°, dip≈30° (shallow E-dipping, includes 359°/1° wrap)
- **Validation holes**: BH-02, BH-07 (when using random holdout with seed=42, fraction=0.25)

## Usage

Run `python generate_m7_demo.py` to regenerate the dataset.

Then in the DFN Cave Studio UI:
1. **Import**: Open M7 Import Wizard → select all 5 files
2. **Clean**: Check quality issues, accept suggested fixes
3. **Holdout**: Select validation boreholes (random, seed=42, 25%)
4. **Domains**: Open Domain Editor, create Domain 1 and Domain 2
5. **Joint Sets**: Choose Mode A (imported set_id) or Mode B (auto, 3 clusters)
