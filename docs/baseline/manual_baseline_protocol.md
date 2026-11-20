# Manual baseline: HR time per annual resume update

**Purpose.** Measure the current manual process so that the IRM time saving is reported against
a measured figure, not an estimate (secondary metric "HR time per resume update").

**Sample.** The 10 resumes listed in `manual_baseline_timesheet.csv` (fixed random sample, seed 42).

**Procedure (per resume).**

1. Start the stopwatch when the HR operator opens the resume file.
2. Record the time for each step in seconds:
   - `t_find_employee_s` – find the employee record in the application UI.
   - `t_read_resume_s` – read the resume.
   - `t_compare_fields_s` – compare the resume with the current record and decide what changed.
   - `t_type_updates_s` – type the changes into the record and save.
3. Record `t_total_s`, the number of fields updated, and (after a second person checks the record) the number of errors found.
4. Use the operator's normal tools. Do not use IRM output during the measurement.

**Data available to size the workload.** From the dataset itself (`workload_stats.json`):
on average 24.6 field values per resume must be read and compared (min 14, max 31), and 4.74 skill-year
values actually change per resume (1,331 changes across 281 resumes; 262 resumes have at least one change).

**Reporting.** Median and interquartile range of `t_total_s`; projected annual hours =
median × number of resumes per cycle. Run by HR.
