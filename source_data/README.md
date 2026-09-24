# Table source data

`Source_Data.xlsx` is the source-data workbook accompanying the revised manuscript, Supplementary Information, and response to reviewers. It has 54 table worksheets and one index. The index identifies table locations, units, and scope. Each worksheet retains the values reported in the corresponding delivery table.

For the shared 160-case classification cohort, PanDerm has 90 correct predictions and 56.25% accuracy; SkinGPT-R1 has 81 correct predictions and 50.63% accuracy. The difference is 9 cases and 5.63 percentage points. The clinician preference study has 158 completed cases and is a separate analysis.

The DermCoT training split contains 15,568 records: 4,394 Light with label 0, 28.22%; 11,170 Medium with label 1, 71.75%; and 4 Dark with label 2, 0.03%. The combined source inventory contains 149,971 Light, 175,957 Medium, and 8,690 Dark records, totalling 334,618 before the internal split and exclusions. These source counts exclude repeated sampling. The four Dark records in DermCoT indicate limited coverage of this group. The updated counts appear in worksheets S2, R1-M5a, and R2-3a.

Full SkinGPT-R1 achieves 69.30% on SkinCon and 42.40% on PAD-UFES-20. These values apply consistently to Full CoT, component Full, and the 8-expert condition. Transcription errors in the corresponding table entries were corrected following confirmation on 24 September 2026. Relative to Direct diagnosis, the Full CoT improvements are 16.67 and 8.81 percentage points, respectively. Other conditions are unchanged.

This workbook provides reported aggregate table data. Raw case-level CSV files and individual clinician assessments are excluded. To recompute aggregate percentages and Wilson intervals, run `python evaluation/check_reported_results.py` from the repository root.
