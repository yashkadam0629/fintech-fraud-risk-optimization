# Reports

This folder is populated by `python run_pipeline.py`. It ships empty on
purpose.

Three files are generated:

| File | Contents |
| --- | --- |
| `findings.md` | Every finding as OBSERVATION / INTERPRETATION / BUSINESS IMPLICATION / RECOMMENDATION |
| `business_recommendations.md` | The strategy layer: intervention policy, where to target friction, threshold governance |
| `FINAL_SUMMARY.md` | Dataset characteristics through to a resume-ready description and interview notes |

Every number in them is read from `outputs/model_results/pipeline_results.json`,
which the pipeline measures on your data. Nothing is typed in by hand, which is
the only reliable way to guarantee that no invented result reaches a report.

So there are no results here until you run the pipeline on the real dataset.
