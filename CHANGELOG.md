# Changelog

Documentation of changes for each version of ``BIDS-Aid``.

**Currently doesn't follow semantic versioning.**

## [0.22.1] - 2026-04-13
- Fix NoneType issue in ``sort_by_acquisition_order`` function

## [0.22.0] - 2026-04-13
- Created new module (``path_utils``)

## [0.21.0] - 2026-04-12
- Splits ``bids`` module to ``files`` and ``events`` modules

## [0.20.0] - 2026-04-12
- Package renamed from ``nifti2bids`` to ``bidsaid`` due to expanding scope. Prior package names deleted from Pypi to allow other users to use those names

## [0.19.0] - 2026-03-10
- Update simulate module

## [0.18.0] - 2026-03-08
- Update to function that parses Presentation log generation date

## [0.17.6] - 2026-03-03
- Add function to get log date for Presentation

## [0.17.5] - 2026-02-22
- Add function to get axis of certain anatomical directions

## [0.17.4] - 2026-02-21
- Adds additional helper functions

## [0.17.3] - 2026-02-20
- Add function to create spike regressors

## [0.17.2] - 2026-02-07
- Prevent dtype warning

## [0.17.1] - 2026-02-02
- Change dictionary key name for global signal function

## [0.17.0] - 2026-02-02
- Add global signal QC metric
- Change return type for compute fd

## [0.16.2] - 2026-02-01
- Forgot to convert the rotation parameters to mm.

## [0.16.1] - 2026-02-01
- Regex glob function now returns a generator instead of list[Path]

## [0.16.0] - 2026-02-01
- Add new QC module
- Change default logging level to "INFO"

## [0.15.0] - 2026-01-31
- Remove the sum durations for ``PresentationBlockExtractor``. Also, note that the previous docs had descriptions in reverse for what was done if the sum durations parameter was true or false

## [0.14.0] - 2026-01-27
- Allow regex to be used in the extractors and in the function that adds instruction
- Change ``rest_block_code`` to ``rest_block_codes``

## [0.13.5] - 2026-01-17
- Increase maxsize for lru cache

## [0.13.4] - 2026-01-17
- Make first level function globbing more flexible

## [0.13.3] - 2026-01-17
- Minor type fix

## [0.13.2] - 2026-01-17
- Add check to ensure analysis dir exists before querying

## [0.13.1] - 2026-01-17
- Add function to ``BIDSAuditor`` to check analysis directory

## [0.13.0] - 2026-01-14
- Allow for more precise block duration times

## [0.12.7.post2] - 2026-01-12
- Fix doc rendering issues

## [0.12.7.post1] - 2026-01-12
- Doc update

## [0.12.7] - 2026-01-11
- Fix ``derivatives_dir`` parameter for ``BIDSAuditor``

## [0.12.6] - 2026-01-10
- Add ``create_sessions_tsv``

## [0.12.5] - 2026-01-10
- Adjust ``BIDSAuditor`` to not produce warning when derivatives are not present

## [0.12.4] - 2026-01-09
- Sort IDs for ``create_participants_tsv``

## [0.12.3] - 2026-01-08
- Add new parameter to ``get_entity_value``

## [0.12.2] - 2026-01-08
- Fix path issue in ``BIDSAuditor``

## [0.12.1] - 2026-01-08
- Allow ``BIDSAuditor`` to handle datasets with no session

## [0.12.0] - 2026-01-07
- Add audit module

## [0.11.4] - 2025-12-31
- Adds warning slice start index

## [0.11.3] - 2025-12-29
- Fix for creating parent directories

## [0.11.2] - 2025-12-29
- Add new bids filename function and create destination directory when creating bids file

## [0.11.1] - 2025-12-28
- Define "__all__"

## [0.11.0] - 2025-12-28
- Move ``get_entity_value`` to bids module

## [0.10.0] - 2025-12-28
- Replace ``glob_contents`` with ``regex_glob``
- Add new parameters to ``compress_image``

## [0.9.1] - 2025-12-27
- Add file timestamp function and return modification date for non-Windows systems

## [0.9.0] - 2025-12-23
- Change ``separate_cue_as_instruction`` to ``split_cue_as_instruction``
- Changed logic for ``response_trial_names`` to not remove rows containing
a trial type of interest for reaction times and accuracy computations.
They will instead be included unless ``split_cue_as_instruction`` used

## [0.8.4] - 2025-12-23
- Fix Window flashing issue with ``convert_edat3_to_text``

## [0.8.3] - 2025-12-22
- Update default for ``convert_eprime_to_text``

## [0.8.2] - 2025-12-22
- Add False for as default for ``response_required_only`` for ``EPrimeBlockExtractor``

## [0.8.1] - 2025-12-21
- Add the ``add_instruction_timing`` function

## [0.8.0] - 2025-12-20
- Several changes and additions made to the BlockExtractor classes, including parameter name changes, removing the ``start_at_cue`` and replacing it with ``separate_cue_as_instruction``, and additional parameters to separate instruction cues from the start stimulus

## [0.7.5] - 2025-12-19
- For `PresentationBlockExtractor`, return the response as recorded by Presentation regardless if participant responded or not

## [0.7.4] - 2025-12-16
- Fix dataframe copy issue

## [0.7.3] - 2025-12-16
- Fix ``convert_edat3_to_text`` not returning path

## [0.7.2] - 2025-12-16
- Update parameter name for create bids file

## [0.7.1] - 2025-12-15
- Add ability for blocks to start at cue or stimulus for onset and duration

## [0.7.0] - 2025-12-15
- Change ``trial_types`` parameters to ``block_cue_codes`` for block extractors
- Compute mean accuracy and reaction times for blocks
- Add accuracy to Presentation event extractor
- Change ``extract_responses`` to ``extract_accuracies`` for E-Prime event extractor
- Fix issue when scanner start time is 0

## [0.6.2.post2] - 2025-12-14
- Clean doc string

## [0.6.2.post1] - 2025-12-14
- Add doc string resource

## [0.6.2] - 2025-12-14
- Datetime fix for older Python versions

## [0.6.1] - 2025-12-14
- New function to get file creation date

## [0.6.0] - 2025-12-13
- For PresentationEventExtractor, make duration the "Duration" column
- For event extractors, create reaction time functions and improve logic
- ``duration_column_name`` changed to ``offset_column_name``

## [0.5.0.post1] - 2025-12-12
- Doc string fix

## [0.5.0] - 2025-12-12
- Add parameter for trigger column for E-Prime classes

## [0.4.0] - 2025-12-10
- Change ``convert_edat3_to_tsv`` to ``convert_edat3_to_text``
- Add a ``format`` parameter to ``convert_edat3_to_text`` to allow tsv and csv
- In ``convert_edat3_to_text``, default to csv

## [0.3.6.post1] - 2025-12-05
- Improve docs

## [0.3.6] - 2025-12-04
- Add parameters for rest code frequency and quit code to block extractors for better duration
computation

## [0.3.5] - 2025-12-03
- Iterate over filtered indices instead of entire dataframe
- In the block extractor for E-Prime, put onset_column_name in init for
consistency

## [0.3.4] - 2025-12-03
- Replace use_first_pulse with scanner_trigger_code and scanner_event_code
- Improve series filtering code

## [0.3.3] - 2025-12-03
- Add new trial column parameter to Presentation classes

## [0.3.2] - 2025-12-03
- More flexibility for log parsing

## [0.3.1.post1] - 2025-12-03
- Add directive to doc

## [0.3.1] - 2025-12-02
- Fix default value issue

## [0.3.0] - 2025-12-02
- Add new parameters to the Presentation Extractor classes

## [0.2.11] - 2025-12-01
- Ensure starting block indices are sorted for EPrimeBlockExtractor

## [0.2.10] - 2025-12-01
- Add new classes and functions for getting
onsets, durations, trial types and responses from
EPrime and Presentation logs

## [0.2.9] - 2025-11-28
- Change `get_date_from_filename` to `parse_date_from_path`

## [0.2.8] - 2025-11-28
- Added assertions

## [0.2.7] - 2025-11-28
- Add function to convert edat3 to tsv

## [0.2.6] - 2025-11-27
- Add new parameter for parser functions

## [0.2.5] - 2025-11-27
- Add E-Prime 3 parser
- Convert the `convert_to_seconds` from bool to a iterable
- Add `convert_to_seconds` to the presentation to events function

## [0.2.4] - 2025-11-25
- Add new function to convert presentation logs to bids event files
- Simplify some function parameters

## [0.2.3] - 2025-11-21
- Add new slice acquisition methods
- Remove ability to use indivisible multiband factor since ordering
may depend on software version for GE.

## [0.2.2] - 2025-11-21
- Add fallback trt parameter

## [0.2.1] - 2025-11-20
- Fix potential edge case in parsing function

## [0.2.0] - 2025-11-20
- Add parser module

## [0.1.9] - 2025-11-19
- Parameter name change for ``infer_task_for_image`` and give ability
to handle different mapping
- Decorator to check if NIfTI image is raw

## [0.1.8] - 2025-11-15
- Return acquisition parameter to slice_acquisition_method

## [0.1.7] - 2025-11-15
- Added new functions and changed accepted values for for parameters relates to
axes (from x, y, z to i, j, k)

## [0.1.6] - 2025-11-14
- Typing and docs fixes, including change for ``create_affine`` to
accept tuples and lists

## [0.1.5] - 2025-11-13
- Doc and type fixes plus accept Path objects

## [0.1.4] - 2025-11-12
- Change parameter name from "interleave_pattern" to "interleaved_pattern"

## [0.1.3] - 2025-11-11
- Add logic for indivisible multiband factor for all cases except for philips
interleaved pattern

## [0.1.2] - 2025-11-11
- Inspired by the following article to write better code: https://pmc.ncbi.nlm.nih.gov/articles/PMC5274797/
    - Essentially add Philip's as an interleave pattern while only retaining sequential or interleave for acquisition
- Some parameter name changing

## [0.1.1] - 2025-11-10
- Add Philip's specific interleaved order and multiband slice acquisition
- Other parameter name changes

## [0.1.0] - 2025-11-06
- Change utils module name to metadata
- Change logger module name to logging
- Create new bids module and move ``create_bids_file``, ``create_dataset_description``, ``save_dataset_description``, and ``create_participant_tsv`` to it
- Add ``save_df`` and ``return_df`` parameters to ``create_participant_tsv``

## [0.0.9] - 2025-11-05
- Add ``slice_axis`` parameter to ``create_slice_timing``

## [0.0.8] - 2025-11-05
- Change function and parameter names ending in "dim" to "axis"
- Change custom exception name
- Add new function to infer task based on number of volumes
- Add level parameter to ``setup_logger``
- Rename package from ``BidsPrep`` to ``Nifti2Bids``

## [0.0.7] - 2025-11-05
- Add  ``get_n_volumes`` function and change custom exceptions names

## [0.0.6] - 2025-11-04
- Add exception to ``create_slice_timing`` for safety

## [0.0.5] - 2025-11-04
- Fix ``create_bids_filename`` to not add "desc"
- Return numeric values as regular Python integers and float
- Add function to extract entity value
- Change ``destination_dir`` and ``output_dir`` to ``dst_dir``

## [0.0.4] - 2025-11-04
- Add function to create participants tsv file.
- ``get_files`` changed to ``glob_contents``.

## [0.0.3] - 2025-11-04
- Add function for extracting date from filenames.

## [0.0.2] - 2025-11-04
- Change output of ``create_slice_timing`` from a dictionary to a list.

## [0.0.1] - 2025-11-03
- First non-alpha release of ``BIDSPrep``.
